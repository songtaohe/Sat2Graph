import common 

def connectDeadends(neighbor, threshold = 30):
    deadends = []
    for k, v in neighbor.iteritems():
        if len(v) == 1:
            deadends.append(k)

    pairs = []
    for i in range(len(deadends)):
        for j in range(i+1, len(deadends)):
            d = common.distance(deadends[i], deadends[j])
            if d < threshold:
                pairs.append((i,j,d))

    pairs = sorted(pairs, key=lambda  x: x[2])
    coverSet = set()

    for pair in pairs:
        if pair[0] not in coverSet and pair[1] not in coverSet:
            n1 = deadends[pair[0]]
            n2 = deadends[pair[1]]

            if n1 not in neighbor[n2]:
                neighbor[n2].append(n1)
            
            if n2 not in neighbor[n1]:
                neighbor[n1].append(n2)

            coverSet.add(pair[0])
            coverSet.add(pair[1])


    return neighbor

