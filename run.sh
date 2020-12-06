question=$1
train_path=$2
test_path=$3
output_path=$4

if [[ ${question} == "1" ]]; then
python q1.py $train_path $test_path $output_path
fi

if [[ ${question} == "2" ]]; then
python q2.py $train_path $test_path $output_path
fi

if [[ ${question} == "3" ]]; then
python q3.py $train_path $test_path $output_path
fi