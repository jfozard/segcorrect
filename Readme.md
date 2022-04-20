
## Segcorrect tool

# Install

Install miniconda / anaconda, then create a new environment with all dependencies:

    conda install -n segcorrect -f env.yml

Precise versions (for linux) given in `linux_env.yml`

# Run


    conda activate segcorrect
    cd src
    python mini_tool.py


At the first file dialogue box select segmented stack, at the second the original (wall) signal stack.
Click to reposition cursor. The "1" key selects the cell under the cursor. "2" adds/removes the cell under the cursor from the current selection. "3" merges all cells in the current selection. Save corrected segmentation using `File > Save as` menu option.
