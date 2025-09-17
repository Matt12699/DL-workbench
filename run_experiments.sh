#!/bin/bash
set -e


# 0.001% delle labels

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/supervised" --labelsRatio 0.00001
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/semi-supervised-autoEncoder-based" --labelsRatio 0.00001
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/semi-supervised-contrastive" --labelsRatio 0.00001
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 0.00001
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.001%labels/semi-supervised-contrastive"


# 0.01% delle labels

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/supervised" --labelsRatio 0.0001
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-autoEncoder-based" --labelsRatio 0.0001
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-contrastive" --labelsRatio 0.0001
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 0.0001
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.01%labels/semi-supervised-contrastive"

# 0.1% delle labels

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/supervised" --labelsRatio 0.001
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/semi-supervised-autoEncoder-based" --labelsRatio 0.001
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/semi-supervised-contrastive" --labelsRatio 0.001
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 0.001
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-0.1%labels/semi-supervised-contrastive"


# 1% delle labels

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/supervised" --labelsRatio 0.01
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/semi-supervised-autoEncoder-based" --labelsRatio 0.01
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/semi-supervised-contrastive" --labelsRatio 0.01
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 0.01
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-1%labels/semi-supervised-contrastive"

# 50% delle labels

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/supervised" --labelsRatio 0.5
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/semi-supervised-autoEncoder-based" --labelsRatio 0.5
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/semi-supervised-contrastive" --labelsRatio 0.5
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 0.5
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-50%labels/semi-supervised-contrastive"

#100% labels

python src/main.py unsupTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/unsupervised"
python src/main.py unsupEvaluate --autoEncoder src/model/unsupervised_trained_autoEncoder.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/unsupervised"

python src/main.py supTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/supervised" --labelsRatio 1
python src/main.py supEvaluate --model src/model/supervised_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/supervised"

python src/main.py AutoEncoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/semi-supervised-autoEncoder-based" --labelsRatio 1
python src/main.py semiSupEvaluate --encoder src/model/autoEncoder_trained_encoder.pth --model src/model/autoEncoder_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/semi-supervised-autoEncoder-based"

python src/main.py encoderTrain --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/semi-supervised-contrastive" --labelsRatio 1
python src/main.py encoderClassTrainContrastive --input resources/datasets/Scaling_ton_Train.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/semi-supervised-contrastive" --encoder src/model/contrastive_trained_encoder.pth --labelsRatio 1
python src/main.py semiSupEvaluate --encoder src/model/contrastive_trained_encoder.pth --model src/model/contrastive_trained_model.pth --input resources/datasets/Scaling_ton_Test.csv --dataset dataset --plotsDir "ton_v3/Scaling-100%labels/semi-supervised-contrastive"