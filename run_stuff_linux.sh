#!/bin/bash

for i in {1..24}
do
    python -u run_ode.py $i > logs/all_testing$i.log 2>&1 &
done

#!/bin/bash



# for i in {1..36}
# do
#     python -u tune_encoders.py $i > output_$i.log 2>&1 &
#     # Capture tqdm output to the log file without taking up lots of lines
#     echo -e "\nRunning process $i\n" >> output_$i.log
#     python -u tune_encoders.py $i 2>&1 | sed "s/^/\t/" >> output_$i.log &
# done
