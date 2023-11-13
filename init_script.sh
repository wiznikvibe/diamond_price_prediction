echo [$(date)]: "START"

echo [$(date)]: "Creating venv with python== 3.8"

conda create --prefix ./venv python=3.8 -y

echo [$(date)]: "Virtual Enviroment Created"
# 1:11:32