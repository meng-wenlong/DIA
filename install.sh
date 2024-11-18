conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers
pip install openai
pip install ollama
pip install JailbreakEval
pip install tenacity
pip install datasets
pip install nltk
pip install language-tool-python
pip install sentence-transformers

# [Optional] For experiments
pip install evaluate

# Install ri
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/src" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh