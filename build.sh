#wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv -O covid-eucdc.csv
#wget https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv/data.csv -O covid-eucdc.csv
wget https://covid.ourworldindata.org/data/owid-covid-data.csv -O covid-eucdc.csv

mkdir -p build
python3 build.py
cp static/* build
