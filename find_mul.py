import torch
from ctcdecode import *
import sys, os, ctcdecode 
from hazm import Normalizer,sent_tokenize



alpha = '۱۲۳۴۵۶۷۸۹۰آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی.،؛:!؟ ‌_>'
farsi_cahr = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
punctuation = '.،؛:!؟'
SPACE_LOCATION = alpha.index(' ')
SEMI_SPACE_LOCATION = SPACE_LOCATION + 1
#blank space is '_' 



steps = lambda:[x / 20 for x in range(20)]

def convert_to_string(tokens, vocab, seq_len):
    return ''.join([vocab[x] for x in tokens[0:seq_len]])

def set_mul(i,j,k,space_indices, sentence_for_beam,ctc_prob):
	for x in space_indices:
		if sentence_for_beam[x] == ' ':
			ctc_prob[x][SPACE_LOCATION] = .5
			ctc_prob[x][SEMI_SPACE_LOCATION] = .3
			ctc_prob[x][-2] = .1
		elif sentence_for_beam[x] == '‌':
			ctc_prob[x][SPACE_LOCATION] = .05
			ctc_prob[x][SEMI_SPACE_LOCATION] = .5
			ctc_prob[x][-2] = .1
		else:
			ctc_prob[x][SPACE_LOCATION] = i
			ctc_prob[x][SEMI_SPACE_LOCATION] = j
			ctc_prob[x][-2] = k
	ctc_prob[-1][-1] = 1


def run_ctcdecoder(decoder, ctc_prob,expected=''):
	probs_seq = torch.FloatTensor([ctc_prob])
	beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
	if expected != '':
		answer = convert_to_string(beam_result[0][0], alpha, out_seq_len[0][0])
		print(clean_punctuation(answer))		
		if expected in answer:
			return True
		return False
	else:
		answer = convert_to_string(beam_result[0][0], alpha, out_seq_len[0][0])
		print(clean_punctuation(answer))
		return True


def search_for_best_mul(decoder,ctc_prob,space_indices,sentence_for_beam,expected_sentence):
	step_counter = 1
	for i in steps():
		for j in steps():
			for k in steps():
				set_mul(i,j,k,space_indices, sentence_for_beam,ctc_prob)
				output_str = run_ctcdecoder(decoder,ctc_prob)
				if expected_sentence == '':
					print(i,j,k,step_counter)
					print(output_str)
				elif expected_sentence in output_str :
					print(i,j,k)
				step_counter += 1


def read_from_file(file_name):
	file_to_read = open(file_name,'r')
	line = file_to_read.readlines()
	mult=([[float(x) for x in y[:-1].split(' ')]for y in line])
	file_to_read.close()
	return mult


def prepare_line_for_search(line, is_concrete=False):
	space_indices = []

	if not is_concrete:
		sentence = ''
		for word in line:
			sentence += word + ' '
	else:
		sentence = line
	sentence_for_beam = ''
	for x in range(len(sentence)):
		if sentence[x] != ' ' and x != len(sentence)-1 and sentence[x+1] != ' ' :
			sentence_for_beam += sentence[x] + '_'
			space_indices.append(len(sentence_for_beam) - 1)
		else:
			sentence_for_beam += sentence[x]
		if sentence[x] == ' ':
			space_indices.append(len(sentence_for_beam) - 1)
	return sentence_for_beam ,space_indices


def make_ctc_matrix(sentence_for_beam):
	return [[1 if alpha[y] == x else 0 for y in range(len(alpha))]for x in sentence_for_beam]


def tokenize(paragraph, wanted_list):
	normal = Normalizer(remove_extra_spaces=True, punctuation_spacing=True, persian_style=False,persian_numbers=False, remove_diacritics=False, affix_spacing=False, token_based=False)
	for sentence in sent_tokenize(normal.normalize(paragraph)):
		wanted_list.append(sentence)

def read_multi_paragraph_text():
	hole_text = []
	while True:
		paragraph = input()
		if len(paragraph) == 0:
			continue
		if paragraph[-1] == '$':
			tokenize(paragraph[:-1],hole_text)
			break
		tokenize(paragraph,hole_text)
	return hole_text


def clean_punctuation(line):
	cleaned_line = ''
	for x in range(len(line)-1):
		if line[x] == ' ' and line[x+1] in punctuation:
			continue
		cleaned_line +=  line[x]
	return cleaned_line

lm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'words.klm')
decoder = ctcdecode.CTCBeamDecoder(alpha, beam_width=20,
			                                   blank_id=alpha.index('_'),
			                                   model_path=lm_path, alpha=.45, beta=3)

#for get input from shell
#line = sys.argv[1:]
#prepare_line_for_search(line)
hole_text = read_multi_paragraph_text()
for paragraph in hole_text:
	sentence_for_beam, space_indices = prepare_line_for_search(paragraph,True)
	ctc_prob = make_ctc_matrix(sentence_for_beam)
	item = read_from_file('result2')[-1]
	set_mul(item[0], item[1], item[2], space_indices, sentence_for_beam, ctc_prob)
	run_ctcdecoder(decoder,ctc_prob)