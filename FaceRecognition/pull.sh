#!/bin/bash

rm out.*
git pull origin master

sbatch runExp.s
