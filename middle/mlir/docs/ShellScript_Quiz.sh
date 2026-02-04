#!/bin/bash
#===----------------------------------------------------------------------===#
# Shell Script Quiz: full_pipeline.sh
#===----------------------------------------------------------------------===#
#
# Fill in the blanks (marked with ???) to complete the script.
# After completing, compare with scripts/full_pipeline.sh
#
# To test your answers:
#   chmod +x docs/ShellScript_Quiz.sh
#   ./docs/ShellScript_Quiz.sh test/add_only.mlir
#
#===----------------------------------------------------------------------===#

# Q1: What command makes the script stop immediately if any command fails?
set -e 

# Q2: Get the directory where this script is located
# Hint: $0 is the script path, dirname gets directory, pwd prints absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Q3: Get the parent directory of SCRIPT_DIR
# Hint: dirname gets parent directory
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Q4: Define LLVM installation path using home directory variable
# Hint: $HOME is the home directory
LLVM_DIR="$(HOME)/llvm-install"

# Q5: Get the first command-line argument
# Hint: $1 is first argument, $2 is second, etc.
INPUT="$1"

# Q6: Check if INPUT is empty and show usage
# Hint: -z tests if string is empty
if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input.mlir>"
  # Q7: Exit with error code (non-zero = error)
  exit 1  
fi

echo "=== Step 1: Gawee -> Linalg ==="

# Q8: Run gawee-opt and capture output into STEP1
# Hint: $(...) captures command output
STEP1=$("$PROJECT_DIR/build/gawee-opt" --convert-gawee-to-linalg "$INPUT")

# Q9: Print the STEP1 variable
# Hint: echo with variable, use quotes
??? "$STEP1"

echo ""
echo "=== Step 2: Bufferize (tensor -> memref) ==="

# Q10: Pipe STEP1 to mlir-opt for bufferization
# Hint: echo "$VAR" | command
STEP2=$(??? "$STEP1" ??? "$LLVM_DIR/bin/mlir-opt" --one-shot-bufferize="bufferize-function-boundaries")

echo "$STEP2"

echo ""
echo "=== Step 3: Linalg -> SCF Loops ==="

# Q11: Pipe STEP2 to mlir-opt for loop conversion
STEP3=$(echo "$STEP2" | "$LLVM_DIR/bin/mlir-opt" ???)

echo "$STEP3"

echo ""
echo "=== Done! ==="

#===----------------------------------------------------------------------===#
# Answer Key
#===----------------------------------------------------------------------===#
#
# Q1:  set -e
# Q2:  cd, pwd
# Q3:  dirname
# Q4:  $HOME
# Q5:  $1
# Q6:  -z
# Q7:  1
# Q8:  $(  )
# Q9:  echo
# Q10: echo, |
# Q11: --convert-linalg-to-loops
#
#===----------------------------------------------------------------------===#
