# This script reads from related.txt and sample_descriptions.txt, filters the unnecessary products and outputs
# train-test split in the form of files containing json objects in each row.

# Step 1: read related.txt and sample_descriptions.txt into dicts
# Step 2: remove unnecessary products

# definition of unnecessary products
# 1. ignore products (from related.txt) that don't appear in sample desc
#   - this may additionally lead to removal of rows from related.txt (when the related products' list of a product becomes empty)
#   which further requires removing the elements from related products' lists. We loop until convergence as in
#   removeEmpty(..).
# 2. ignore products (from related.txt and sample_descriptions.txt) that are not labels of any product
# 3. ignore products (from related.txt and sample_descriptions.txt) that don't have any labels
#   - this may additionally lead to removal of rows from related.txt (when the related products' list of a product becomes empty)
#   which further requires removing the elements from related products' lists. We loop until convergence as in
#   type3Removal(..).

# Note that we remove the products as defined above since they don't give any information to the learning process.

# Step 3: create a list containing json objects, each representing one product ({"id": productId, "description": product description, "related_products" : [indices of related products]})
# Step 4: split this list to get the train and test sets
# Step 5: write these sets to files

from copy import deepcopy
import pickle, random, json

def readRelated():
    dd = {}
    filename = "./related.txt"
    with open(filename, "r") as f:
        for line in f:
            row = line[:-1].split()
            dd[row[0]] = row[3:]
    return dd

def readSampleDesc():
    filename = "./sample_descriptions.txt"
    dd = {}
    row = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            rem = i % 3
            if rem == 0:
                row = line[:-1].split("product/productId: ")[1]
            elif rem == 1:
                dd[row] = line[:-1].split("product/description: ")[1]
            else:
                continue
    return dd

def countEmpty(relcopy):
    c=0
    for i in relcopy:
        if relcopy[i] == []:
            c+=1
    return c

def removeEmpty(relcopy):
    while True:
        if(countEmpty(relcopy)) > 0:
        # relcopy = {key: relcopy[key] for key in relcopy if relcopy[key] != []}
            removed = set()
            related = {}
            for item in relcopy:
                if relcopy[item] == []:
                    removed.add(item)
                else:
                    related[item] = relcopy[item]

            # remove the removed items from related lists
            for item in related:
                related[item] = [prod for prod in related[item] if prod not in removed]

            relcopy = related
        else:
            return relcopy

def filterProducts(related, sampleDesc):
    relcopy = deepcopy(related)
    # ignore products which don't appear in sample desc

    # Removing an element from list
        # list.remove(element) <- removes the first matching element (which is passed as an argument) from the list.
        # del list[index]

    # Removing an element from dict
        # del dict[key]

    print("\nType 1 removal")
    for item in related:
        if item not in sampleDesc:
            print("Removing entry {} from related".format(item))
            del relcopy[item]
        else:
            for i, prod in enumerate(related[item]):
                if prod not in sampleDesc:
                    print("Removing list item {} from some related list".format(prod))
                    relcopy[item].remove(prod)  # <-- creates rows with empty related items

    # remove keys with empty values
    relcopy = removeEmpty(relcopy)

    # relcopy = {key: relcopy[key] for key in relcopy if relcopy[key] != []}

    del related
    related = relcopy
    relcopy = deepcopy(related)
    print("\nAfter type 1 removal")
    print("related", len(related))
    print("sample desc", len(sampleDesc))

    # ignore products which are not labels of any product
    print("\nType 2 removal")
    labels = set()
    for item in related:
        for prod in related[item]:
            labels.add(prod)

    sampleDescCopy = deepcopy(sampleDesc)
    for item in sampleDesc:
        if item not in labels:
            print("Removing entry {} from related and sample desc".format(item))
            del sampleDescCopy[item]
            if item in relcopy:
                del relcopy[item]

    del related
    related = relcopy
    relcopy = deepcopy(related)
    del sampleDesc
    sampleDesc = sampleDescCopy
    sampleDescCopy = deepcopy(sampleDesc)
    print("\nAfter type 2 removal")
    print("related", len(related))
    print("sample desc", len(sampleDesc))

    # ignore a product if it doesn't have any labels
    print("\nType 3 removal")
    related, sampleDescCopy = type3Removal(related, sampleDescCopy)

    del sampleDesc
    sampleDesc = sampleDescCopy
    sampleDescCopy = deepcopy(sampleDesc)
    print("\nAfter type 3 removal")
    print("related", len(related))
    print("sample desc", len(sampleDesc))

    return related, sampleDesc

def deleteFromRelated(related, toRemoveFromRelated):
    for item in related:
        related[item] = [prod for prod in related[item] if prod not in toRemoveFromRelated]

def getToRemoveFromRelated(related, sampleDescCopy):
    toRemoveFromRelated = set()
    for item in sampleDescCopy:
        if item not in related:
            toRemoveFromRelated.add(item)
            print("Removing entry {} from sample desc".format(item))
            # del sampleDescCopy[item]
    return toRemoveFromRelated

def type3Removal(related, sampleDescCopy):
    toRemoveFromRelated = getToRemoveFromRelated(related, sampleDescCopy)
    while len(toRemoveFromRelated) > 0:
        # delete from sample desc
        for item in toRemoveFromRelated:
            del sampleDescCopy[item]
        # delete from internal lists in related
        deleteFromRelated(related, toRemoveFromRelated)
        # delete from related until convergence
        related = removeEmpty(related)
        toRemoveFromRelated = getToRemoveFromRelated(related, sampleDescCopy)
    return related, sampleDescCopy

def getProdList(relatedList, productIdToIdx):
    ret = []
    for prod in relatedList:
        ret.append(productIdToIdx[prod])
    return ret

def jsonDump(related, sampleDesc, productIdToIdx):
    """ creates a list of json objects of type:
        {
            "id": productId,
            "description": product description,
            "related_products" : [indices of related products]
        }
    """
    listOfJson = []
    for item in sampleDesc:
        jsonObj = {
            "id": item,
            "description": sampleDesc[item],
            "related_products": getProdList(related[item], productIdToIdx)
        }
        listOfJson.append(json.dumps(jsonObj))

    return listOfJson

def writeJsonListToFile(filename, listOfJson):
    with open(filename, "w") as f:
        for elem in listOfJson:
            print(elem, file=f)

def main():
    related = readRelated()
    sampleDesc = readSampleDesc()
    related, sampleDesc = filterProducts(related, sampleDesc)
    print("\nin main...")
    print("related", len(related))
    print("sample desc", len(sampleDesc))
    productIdToIdx = {}
    for i, item in enumerate(sampleDesc):
        productIdToIdx[item] = i
    listOfJson = jsonDump(related, sampleDesc, productIdToIdx)
    writeJsonListToFile("./filtered.txt", listOfJson)

    with open("filtered_json.pkl", "wb") as f:
        pickle.dump(listOfJson, f)

    # with open("filtered_json.pkl", "rb") as f:
    #     listOfJson = pickle.load(f)

    # split into train and test and write to separate files
    train, test = getMaxLabelCoverageSplit(listOfJson)

    writeJsonListToFile("trn.json", train)
    writeJsonListToFile("tst.json", test)

    print()

def getRandomSplit(listOfJson):
    random.seed(25000)
    random.shuffle(listOfJson)
    lim = int(0.8 * len(listOfJson))
    train = listOfJson[:lim]
    test = listOfJson[lim:]
    return train, test

def getMaxLabelCoverageSplit(listOfJson):
    """
    Splits are created in a way that ensures that maximum labels have at least one training point.
    """
    labelLen = []
    for i, item in enumerate(listOfJson):
        item = json.loads(item)
        labelLen.append((i, len(item["related_products"]), item["related_products"]))
    labelLen.sort(key=lambda x: x[1], reverse=True)

    indices_needed = int(0.8 * len(listOfJson))
    labelCovered = set()
    indices_chosen = []
    indices_remaining = [i for i in range(len(listOfJson))]
    for tup in labelLen:
        if(len(indices_chosen) < indices_needed and set(tup[2]).isdisjoint(labelCovered)):
            labelCovered.update(tup[2])
            indices_chosen.append(tup[0])
            indices_remaining.remove(tup[0])

    if(len(indices_chosen) < indices_needed):
        # need to choose more
        toChoose = indices_needed - len(indices_chosen)
        indices_chosen.extend(indices_remaining[:toChoose])
        indices_remaining = indices_remaining[toChoose:]

    assert len(indices_chosen) == indices_needed
    assert len(indices_remaining) == len(listOfJson) - indices_needed

    print("Total labels covered:",len(labelCovered))

    train = [item for i, item in enumerate(listOfJson) if i in set(indices_chosen)]
    test = [item for i, item in enumerate(listOfJson) if i in set(indices_remaining)]

    return train, test


if __name__ == "__main__":
    main()