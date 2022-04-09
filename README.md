# Fun with Wordle
The file `wordle.py` contains utility functions and code for optimizing the Wordle strategy in a tree structure.

The file `wordlePlayer.ipynb` solves an actual Wordle game by asking the user to make certain guesses and report back the hints.

Uses the word lists found here: https://github.com/3b1b/videos/tree/b1ad11dfb3d38a9f5d3f1c9e4548208be51e1b96/_2022/wordle/data.

## Strategy
In short, I started out with utilizing information gain as in [3Blue1Brown's YouTube video](https://youtu.be/v68zYyaEmEA). I computed the top 10 1-step starting words by this metric. Then, for each of these words and for each possible hint on the first guess, I computed the top 10 words for the next guess in the same way. Continuing with this allows for creating a tree of "good" Wordle games. This is useful because 1) it is much smaller than the tree of all possible Wordle games, and 2) there's still a good chance it contains the optimal strategy. It is then easy to trim the tree down to the Wordle strategy that achieves the lowest average number of gueses (I got 3.4205).
