'''This code will take an input word as in string and will
output the PHOC label of the word. The Phoc label is a
vector of length 3784.
((2 + 3 + 4 + 5) * languageCharactersAndNumbersCount) + (2*commonBigram)
((2+3+4+5) * 256) + (2*100) = 3784
((2+3+4+5) * 45) + (2*50) = 730
Reference: https://ieeexplore.ieee.org/document/6857995/?part=1
'''

def generate_45(word):
  '''The vector is a binary and stands for:
  https://en.wikipedia.org/wiki/Arabic_script_in_Unicode
  arabic unicode characters is 256
  '''
  generate_45 = [0 for i in range(45)]
  for char in word:
      generate_45[ord(char) - ord('ا')] = 1

  return generate_45

def generate_50(word):
  '''This vector is going to count the number of most frequent
  bigram words found in the text
  '''

  bigram = ['لم', 'لل', 'ين', 'لت', 'لي', 'يت', 'لع', 'هم', 'لن', 'تم', 'في', 'عل',
            'لب', 'ست', 'بي', 'يم', 'مت', 'ته', 'لح', 'لق', 'ما', 'لف', 'من', 'ها',
            'له', 'كم', 'يس', 'مل', 'بت', 'لك', 'نا', 'لس', 'يب', 'بع', 'مس', 'سب',
            'يع', 'تح', 'يل', 'فت', 'فل', 'مع', 'تع', 'لا', 'تن', 'تب', 'يح', 'يه',
            'لج', 'فع']

  vector_50 = [0 for i in range(50)]
  for char in word:
    try:
      vector_50[bigram.index(char)] = 1
    except:
      continue

  return vector_50

def phoc_generate_label(word):
  word = word.lower()
  vector = []
  L = len(word)
  for split in range(2, 6):
    parts = L//split
    for mul in range(split-1):
      vector += generate_45(word[mul*parts:mul*parts+parts])
    vector += generate_45(word[(split-1)*parts:L])

  # Append the most common 50 bigram text using L2 split
  vector += generate_50(word[0:L//2])
  vector += generate_50(word[L//2: L])
  return vector
