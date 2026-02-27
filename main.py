import csv

with open("votes.csv", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
