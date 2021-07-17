# DataDrivenModelling
The data-sets were too large to upload to GitHub


Running the code is done from the run.py file.

Changing the setup of the neural network is done in the config.sjon file

The way the code functions, and the theory behind how to use it can be found in my Thesis 

A quick guide on how to use the LSTM-code is:
1. Put your dataset in the data-folder
2. Use the normalize_dataset function with the absolute path of the file as the argument
3. Check the config.json file, alter it such that the setup is correct. (Remember that "input_dim" in the first layer should be equal to the number of columns, and "input_timestep" should be equal to "sequence_length"-1)
4. The two lines "old=sys.stdout" and "sys.stdout=open("engine.txt",'w')" changes the console output to be a textfile, so alter the name of the textfile to the desired filename
5. Run the main() function
6. Assess how the neural network performed and do changes as necessary


