def replace_reserved(string):
    res = ""
    for c in string:
        if c != '^':
            res += c
    string = res
    
    index_dict = {}
    result = []

    for i, word in enumerate(word_list):
        inde = string.find(word)
        while(inde > -1):
            string = string[:inde] + '^' * len(word)+ string[inde + len(word):]
            index_dict[inde] = i
            inde = string.find(word)
    
    cnt = 0
    for i, char in enumerate(string):
        if cnt == 0:
            if char != '^':
                result.append(char)
            else:
                word = word_list[index_dict[i]]
                cnt = len(word) - 1
                result.append(word)
        else:
            cnt -= 1;
            
    return list(result)
