cd /media/DualTopoDDI

echo "Start Inferencing ..."
echo "##################################################################################################"
echo "\n"
echo "For drugbank:"

python test.py


echo "For MMDDI and DDInter datasets:"

python test_MMDDI.py

echo "For AUC_FC and AUC_FC_External datasets:"

python test_AUC_FC.py

wait
echo "********************************************************"
echo "\n"

echo "##################################################################################################"
echo "END ..."

