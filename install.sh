# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
pip3 install packaging -i https://pypi.org/simple
pip3 install numpy==1.26.4 -i https://pypi.org/simple
pip3 install torch==2.4.0 -i https://pypi.org/simple
pip3 install triton==3.0.0 -i https://pypi.org/simple
pip3 install transformers==4.44.0 -i https://pypi.org/simple
pip3 install Cython==3.0.11 -i https://pypi.org/simple
pip install nemo-toolkit[all]==1.21 --no-deps -i https://pypi.org/simple
pip3 install -r extra_requirements.txt -i https://pypi.org/simple