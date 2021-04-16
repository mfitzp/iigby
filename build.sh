wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv -O covid-eucdc.csv

mkdir -p build
python3 build.py
cp static/* build