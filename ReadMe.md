# Omnifold Plot



This project was made using tools and data from ACU Atom Smashers, in the summer of 2025, by William Coker



## Installation



Run this command in the desired installation folder:

```bash
git clone https://github.com/wdc756/OmnifoldPlot.git
```

Then download the relevant omnifold training data from the Atom Smashers drive, usually found here:
1. [nat/syn data](https://drive.google.com/drive/folders/1k3N68xNQYAaD2Jk2x_XxkEr7aiLZPoFJ?usp=sharing)
2. [Omnifold weights](https://drive.google.com/drive/folders/16vlWNSkjbIgR1CQELLMuM5Nbru6kg3VY?usp=sharing)
3. [Omnifold re-weights](https://drive.google.com/drive/folders/1dBp0jouXa-jANzpoMZaIvvr-19lUQRYB?usp=sharing)
4. [Omnifold spline-weights](https://drive.google.com/drive/folders/10IQoQTLWLmPtGu70ObfWDo6yZQSkj4Z0?usp=sharing)

After that, if you want to use the default plotting options, you'll need to create a folder
```data``` with the following subfolders in your project root:

```text
data
├── mock - (nat/syn)
├── weights - (Omnifold weights)
├── re_weights - (Omnifold re_weights)
├── spline_weights - (Omnifold spline-weights)
└── plots - (to be filled)
```



## Usage



### Default Mode

After setting up your local files using the structure above, you can run the program by calling:

```bash
python /src/omnifold_plot.py
```

To change what it outputs, open ```omnifold_plot.py```, and change the ```average_*``` 
boolean flags. *Note that averaging over tests is always on by default.

### Manual Mode

If you would like more control, such as not averaging tests, changing colors, or adding/removing
datapoints, then set ```use_default_options = False``` and scroll down. All important plotting
information is contained in the comments above each option