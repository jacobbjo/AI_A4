i = [[[] for col in range(10)] for row in range(5)]
i[2][3].append(4)
for row in i:
    print(row)