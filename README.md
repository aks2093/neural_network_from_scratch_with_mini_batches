"# neural_network_from_scratch_with_mini_batches" 


This python file has following oprion to run:
1. ("-b", "--batch_size", help="specify batch size", default="64", required=False)
2. ("-s", "--input_shape", help="specify input shape  separated by ',' ", required=True)
3. ("-n", "--epochs", help="number of epochs", required=False, default=10)
4. ("-l", "--layer_dim", help="layer dimensions separated by ',' ", required=True)
5. ("-r", "--learning_rate", help="learning rate for the network",default=0.001, required=False)

How to run the file:
1. go to file folder
2. run the following command:


python backpropagation_forwardpropagation.py -s <> -l <>

other options can also be specified but they are not required, but -s and -l must be specified and values must be comma separated

Example:
>python backpropagation_forwardpropagation.py -s 1000,100 -l 50,10,1


Note: As of now it supports any 2D data matrix, suppport for more dimensional data matrix can be given by adding
a simple reshaping function.
