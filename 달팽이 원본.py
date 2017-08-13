snail_size = int(input('달팽이 껍질의 크기를 정하시오 '))
# snail_size = 8
snail = [[0 for number in range(1, snail_size + 1)] for number in range(snail_size)]
# for i in range(1, 4)

#####################################

for j in range(snail_size - 2):
    snail[1][j + 1] = snail_size + 1 + (snail_size - 1) * 2 + (snail_size - 2) * 1 + j  # 3n - 4

for j in range(snail_size - 4):
    snail[2][j + 2] = snail_size + 1 + (snail_size - 1) * 2 + (snail_size - 2) * 2 + (snail_size - 3) * 2 + (snail_size - 4) * 1 + j  # 7n - 16

for i in range(snail_size // 2):
    for j in range(snail_size - 2 * (i + 1)):
        snail[i + 1][j + 1 + i] = snail_size + 1 + (3 + 4 * (i)) * snail_size - int(4 + (8 * 1 / 2 * i * (i + 1)) + (4 * i)) + j

for i in range(snail_size // 2):
    for j in range(snail_size - 2 * (i + 1)):
        snail[i + 1][j + 1 + i] = 1

###############################

for j in range(1, snail_size):
    snail[j][snail_size - 1] = snail_size + j  # 0n - 0

for j in range(1, snail_size - 2):
    print(j)
    snail[j + 1][snail_size - 2] = snail_size + j + (snail_size - 1) * 2 + (snail_size - 2) * 2  # 4n - 6

for i in range(snail_size // 2):
    for j in range(1, snail_size - 2 * i):
        snail[j + i][snail_size - 1 - i] = snail_size + j + (0 + 4 * i) * snail_size - int(0 + (8 * 1 / 2 * i * (i + 1)) + (-2 * i))

for i in range(snail_size // 2):
    for j in range(1, snail_size - 2 * i):
        snail[j + i][snail_size - 1 - i] = 2

###########################################################

for j in range(1, snail_size - 1):
    print(j)
    snail[j][0] = snail_size + (snail_size - 1) * 2 + (snail_size - 2) * 1 - (j - 1)  # 3n - 4

for j in range(2, snail_size - 1):
    print(j)
    snail[j][1] = snail_size + 1 + (snail_size - 1) * 2 + (snail_size - 2) * 2 + (snail_size - 3) * 2 + (snail_size - 4) * 1 - (j - 1)  # 7n - 16

for i in range(snail_size // 2):
    for j in range(1 + i, snail_size - 1):
        snail[j][i] = snail_size + i + (3 + 4 * i) * snail_size - int(4 + (8 * 1 / 2 * i * (i + 1)) + (4 * i)) - (j - 1)

for i in range(snail_size // 2):
    for j in range(1 + i, snail_size - 1):
        snail[j][i] = 4

#######################################################

for j in range(snail_size, 1, -1):
    print(j)
    snail[snail_size - 1][snail_size - j] = snail_size + (snail_size - 1) * 2 - (snail_size - j) * 1  # 2n - 2

for j in range(snail_size, 3, -1):
    print(j)
    snail[snail_size - 2][snail_size - j + 1] = snail_size + (snail_size - 1) * 2 + (snail_size - 2) * 2 + (snail_size - 3) * 2 - (snail_size - j) * 1  # 6n - 12

for i in range(snail_size // 2):
    for j in range(snail_size, 1 + 2 * i, -1):
        snail[snail_size - 1 - i][snail_size - j + i] = snail_size + (2 + 4 * i) * snail_size - int(2 + (8 * 1 / 2 * i * (i + 1)) + (2 * i)) - (snail_size - j) * 1

for i in range(snail_size // 2):
    for j in range(snail_size, 1 + 2 * i, -1):
        snail[snail_size - 1 - i][snail_size - j + i] = 3

####################################################
for ptn in snail:
    print(ptn)