
data = {49: {54: [[51, 65, 70], 12]}, 7: {3: [[40, 41], 7]}, 58: {54: [[51], 2]}, 63: {61: [[32], 1], 54: [59, 1]}}


entry1 = data[63]
max_key = max(entry1.keys(), key=lambda key: (entry1[key][1], sum(entry1[key][0])/len(entry1[key][0])))

print("Key with the highest value at index 1 in entry1:", max_key)



