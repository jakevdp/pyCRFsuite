crfsuite learn --model='output/model.dat' 'example_files/train_small.txt'
crfsuite tag -r -p -i --model='output/model.dat' 'example_files/test_small.txt'
