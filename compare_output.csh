csh crfsuite_test.csh > output/out_c.txt
tail -1026 output/out_c.txt > output/out_c.txt

python test.py > ouput/out_python.txt
tail -1026 output/out_python.txt > output/out_python.txt

echo "Differences between crfsuite tagging and python tagging:"
echo ""
diff output/out_c.txt output/out_python.txt
echo "--------------------------------------------------------"
