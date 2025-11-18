pip install transformers
pip install openai
pip install anthropic
pip install ollama
pip install JailbreakEval
pip install tenacity
pip install datasets
pip install nltk
pip install language-tool-python
pip install sentence-transformers
pip install evaluate

# Install di
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/src" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh