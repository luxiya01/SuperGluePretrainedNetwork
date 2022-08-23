DESCRIPTORS="rootsift sift hardnet hardnet8 hynet mkdd sosnet tfeat"
NETWORK="small_gnn"

for DESCRIPTOR in $DESCRIPTORS; do
    echo "=========================="
    echo "Train $DESCRIPTOR using $NETWORK..."
    python cli_train.py fit --config configs/config_template.yaml \
        --config configs/descriptors/$DESCRIPTOR.yaml \
        --config configs/networks/$NETWORK.yaml > 220823-$DESCRIPTOR-$NETWORK.out
    echo "=========================="
    echo "\n\n\n\n\n"
done
