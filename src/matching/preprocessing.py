import unidecode, re, string

def remove_accents(text):
    unaccented_string = unidecode.unidecode(text)
    return unaccented_string
    

def remove_special_characters(text):
    chars = re.escape(string.punctuation)
    
    clean = text.replace('&', 'and')
    clean = clean.replace('a(c)', 'e')
    clean = re.sub(r'[^a-zA-Z0-9 \(\)\n\.]', '', clean)
    clean = re.sub(r' +', ' ', clean)
    return clean


def preprocessing(text):
    clean = text.lower()
    clean = remove_accents(clean)
    clean = remove_special_characters(clean)
    return clean

def to_raw_title(titles, year=False):
    raw = []
    for title in titles:
        try:
            idx1 = title.index(' (')
            aux = title[:idx1]
            if (year):
                idx2 = title.rfind(' (')
                aux = title if (idx1 == idx2) else title[:idx1] + title[idx2:]
            aux = preprocessing(aux)
            raw.append(aux.lower())
        except:
            raw.append(title.lower())
    return raw
