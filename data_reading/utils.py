import re
import csv
import num2words as n2w
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def entry_stem(entry):
    tokenized_entry = word_tokenize(entry)
    stemmer = PorterStemmer()
    stemmed_entry = ""
    for word in tokenized_entry:
        stemmed_entry = stemmed_entry + stemmer.stem(str(word)) + ' '
    return stemmed_entry.strip()


def entry_lemma(entry):
    tokenized_entry = word_tokenize(entry)
    lemmatizer = WordNetLemmatizer()
    lemmatized_entry = ""
    for word in tokenized_entry:
        lemmatized_entry = lemmatized_entry + lemmatizer.lemmatize(str(word)) + ' '
    return lemmatized_entry.strip()


def clean_non_alphanumeric(entry):
    sanitize = lambda word: re.sub(r'[^0-9A-Za-z]+', ' ', word).strip()
    tokenized_entry = word_tokenize(entry)
    return ' '.join(list(map(lambda word: sanitize(word), tokenized_entry)))


def prepare_currencies():
    currencies = {'$': 'dollars', '€': 'euros', '£': 'pounds'}

    with open('../data/currencies.csv', 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        for row in currencies.items():
            writer.writerow(row)


def currenciate(entry):
    currency_dict = {}

    with open('../data/currencies.csv', encoding="utf8") as f_in:
        lines = (line.rstrip('\n') for line in f_in if line.rstrip('\n'))
        symbols = []

        for line in lines:
            symbol, currency = line.split(',')
            currency_dict[symbol] = currency
            symbols.append(symbol)

    currency_regex = "([" + '|'.join(symbols) + "])"

    extracted_currencies = re.split(currency_regex, entry)
    translated = [currency_dict[word] if word in currency_dict else word for word in extracted_currencies]
    joined = ' '.join(translated)

    return re.sub(r" +", " ", joined)


def numbers_to_words(entry):
    numbers = re.findall('[-+]?\d*\.\d+|\d*,\d+|\d+', entry)
    for n in numbers:
        convertible = n.replace(",", ".")
        entry = entry.replace(n, " " + n2w.num2words(convertible) + " ", 1)

    return re.sub(r" +", " ", entry)
