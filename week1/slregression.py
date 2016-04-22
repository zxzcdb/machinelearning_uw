import csv

#source ~/ "test27" /bin/activate

with open('kc_house_train_data.csv','rb') as csvfile:
        cdata = csv.reader(csvfile)
        for row in cdata:
            print ', '.join(row)
