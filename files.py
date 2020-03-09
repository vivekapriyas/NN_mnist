import numpy as np

def write_spiral_to_file(filename = 'spiral_text_file.txt', n = 200):
    features, labels = get_data_spiral_2d(n)
    try:
        file = open(filename, 'x')
    except FileExistsError:
        file = open(filename, 'w')
    features_0 = ''
    features_1 = ''
    labels_str = ''
    #Itererer gjennom nesten hele datasettet, fordi siste element skal håndteres litt annerledes
    for n in range(len(features[0]) - 1):
        features_0 += str(features[0][n]) + ', '
        features_1 += str(features[1][n]) + ', '
        labels_str += str(labels[n][0]) + ', '
    features_0 += str(features[0][len(features[0]) - 1]) + '\n'
    features_1 += str(features[1][len(features[0]) - 1]) + '\n'
    labels_str += str(labels[len(features[0]) - 1][0]) + '\n'
    file.write(features_0)
    file.write(features_1)
    file.write(labels_str)
    file.close()
    
def read_spiral_from_file(filename = 'spiral_text_file.txt'):
    #Denne funksjonen leser en fil som er lagret på formatet fra write_spiral_to_file
    file = open(filename, 'r')
    #Vet at filen lagres med tre linjer
    features_0 = file.readline()
    features_0.strip()
    features_list_0 = features_0.split(', ')
    features_1 = file.readline()
    features_1.strip()
    features_list_1 = features_1.split(', ')
    labels_str = file.readline()
    labels_list = labels_str.split(', ')
    for i in range(len(labels_list)):
        features_list_0[i] = float(features_list_0[i])
        features_list_1[i] = float(features_list_1[i])
        if labels_list[i] == 'True':
            labels_list[i] = True
        else:
            labels_list[i] = False
    return(np.array([features_list_0, features_list_1]), np.array(labels_list))


def writeParams(W_k, b_k, omega, my, filename = 'trainingParams.txt'):
    try:
        file = open(filename, 'x')
    except FileExistsError:
        file = open(filename, 'w')

    Wk_str = ""
    for W in W_k:
        for w in W:
            for i in w:
                Wk_str += str(i) + ','
    Wk_str += '\n'

    bk_str = ""
    for B in b_k:
        for b in B:
            bk_str += str(b) + ','
    bk_str += '\n'

    omega_str = ""
    for o in omega:
        omega_str += str(o) + ','
    omega_str += '\n'

    my_str = str(my[0]) + '\n'

    file.write(Wk_str+bk_str+omega_str+my_str)
    file.close()

def readParams(K=20,d=2, filename = 'trainingParams.txt'):
    try:
        file = open(filename, 'r')
    except FileExistsError:
        print("Kunne ikke finne",filename)

    w_k = np.zeros((K, d, d))
    b_k = np.zeros((K, d))
    omega = np.zeros(d)
    my = np.zeros(1)

    W = file.readline().split(',')
    B = file.readline().split(',')
    O = file.readline().split(',')
    M = file.readline()

    file.close()

    for k in range(K):
        for w in range(d):
            for i in range(d):
                w_k[k][w][i]= float(W[0])
                W.pop(0)

    for k in range(K):
        for i in range(d):
            b_k[k][i]=float(B[0])
            B.pop(0)


    for i in range(d):
        omega[i] = float(O[i])

    my[0] = float(M)

    return w_k,b_k,omega,my


