from csv import DictReader
import os

class DataSet():
    def __init__(self, name="train", path=os.path.dirname(__file__)):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        ct = 0
        for s in self.stances:
            s['Stance ID'] = ct
            ct += 1
            s['Body ID'] = int(s['Body ID'])
            s['Predict'] = '?'

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    def get_labelled_stances(self):
        return [s for s in self.stances if s['Predict'] != '?']

    def get_unlabelled_stances(self):
        return [s for s in self.stances if s['Predict'] == '?']

    def read(self,filename):
        rows = []
        p = os.path.join(self.path, "data")
        p = os.path.join(p, filename)
        with open(p, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
