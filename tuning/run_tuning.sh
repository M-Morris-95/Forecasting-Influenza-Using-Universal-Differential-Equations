#!/bin/bash

for i in {1..36}
do
    python -u tune_encoders.py $i > output_$i.log 2>&1 &
done

