for i in {1..20}
do
    wget http://opendata.cern.ch/record/301/files/dimuon-Jpsi_${i}.ig .
done

for i in {1..20}
do
    mv dimuon-Jpsi_${i}.ig dimuon-Jpsi_${i}.ig.zip
    unzip dimuon-Jpsi_${i}.ig.zip
done