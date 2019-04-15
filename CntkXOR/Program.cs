using System;
using System.Collections.Generic;
using System.Linq;

using CNTK;

namespace CntkXOR
{
    class Program
    {
        static Random random = new Random();

        static Function FullyConnectedLinearLayer(Variable inputs, int outputDim, DeviceDescriptor device, string outputName = "")
        {
            System.Diagnostics.Debug.Assert(inputs.Shape.Rank == 1);
            int inputDim = inputs.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, (uint)random.Next(1, int.MaxValue / 2)),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, inputs, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }

        static Function CreateModel(Variable inputs, int hiddenLayers, int outputDim, DeviceDescriptor device, string outputName = "")
        {
            Function dense1 = FullyConnectedLinearLayer(inputs, hiddenLayers, device, "inputLayer");
            Function tanhActivation = CNTKLib.Tanh(dense1, "hiddenLayer");
            Function dense2 = FullyConnectedLinearLayer(tanhActivation, outputDim, device, "outputLayer");
            var model = CNTKLib.Sigmoid(dense2, outputName);
            return model;
        }

        static Trainer CreateModelTrainer(Function model, Variable inputs, Variable labels)
        {
            var trainingLoss = CNTKLib.BinaryCrossEntropy(new Variable(model), labels, "lossFunction");
            var prediction = CNTKLib.ReduceMean(CNTKLib.Equal(labels, CNTKLib.Round(new Variable(model))), Axis.AllAxes()); // Keras accuracy metric

            // set per sample learning rate
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.1, 1);
            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(model.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(model, trainingLoss, prediction, parameterLearners);

            return trainer;
        }

        static void TrainFromArrays(Trainer trainer, Variable inputs, Variable labels, DeviceDescriptor device, int epochs = 1000, int outputFrequencyInMinibatches = 50)
        {
            int i = 0;

            List<(float[], float[])> dataset = new List<(float[], float[])>()
            {
                (new float[] { 0f, 0f }, new float[] { 0f }),
                (new float[] { 1f, 0f }, new float[] { 1f }),
                (new float[] { 0f, 1f }, new float[] { 1f }),
                (new float[] { 1f, 1f }, new float[] { 0f })
            };

            while (epochs >= 0)
            {
                dataset.Sort((a, b) => random.Next(2) * 2 - 1);
                var miniBatch = new Dictionary<Variable, MinibatchData>()
                {
                    { inputs, new MinibatchData(Value.CreateBatch(inputs.Shape, dataset.SelectMany(d => d.Item1), device), 4, 4, false) },
                    { labels, new MinibatchData(Value.CreateBatch(labels.Shape, dataset.SelectMany(d => d.Item2), device), 4, 4, false) }
                };
                trainer.TrainMinibatch(miniBatch, device);

                PrintTrainingProgress(trainer, i++, outputFrequencyInMinibatches);
                epochs--;
            }
        }

        static void TrainFromMiniBatchFile(Trainer trainer, Variable inputs, Variable labels, DeviceDescriptor device, int epochs = 1000, int outputFrequencyInMinibatches = 50)
        {
            int i = 0;

            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[] { new StreamConfiguration("features", inputs.Shape[0]), new StreamConfiguration("labels", labels.Shape[0]) };
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource("XORdataset.txt", streamConfigurations, MinibatchSource.InfinitelyRepeat, true);

            while (epochs >= 0)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(4, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { inputs, minibatchData[minibatchSource.StreamInfo("features")] },
                    { labels, minibatchData[minibatchSource.StreamInfo("labels")] }
                };
                trainer.TrainMinibatch(arguments, device);

                PrintTrainingProgress(trainer, i++, outputFrequencyInMinibatches);
                if (minibatchData.Values.Any(a => a.sweepEnd))
                    epochs--;
            }
        }

        static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                Console.WriteLine($"Minibatch Epoch: {minibatchIdx,5} loss = {trainLossValue:F6}, acc = {evaluationValue}");
            }
        }

        static void TestPrediction(Function model, DeviceDescriptor device)
        {
            Console.WriteLine();
            Console.WriteLine("Prediction");

            Variable inputVar = model.Arguments.Single();
            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(inputVar.Shape, new float[] { 0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f }, device);
            inputDataMap.Add(inputVar, inputVal);
            var inputData = inputVal.GetDenseData<float>(inputVar);

            Variable outputVar = model.Output;
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            model.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            var outputVal = outputDataMap[outputVar];
            var outputData = outputVal.GetDenseData<float>(outputVar);

            for (int k = 0; k < 4; ++k)
                Console.WriteLine("[{0}] => {1}", string.Join(" ", inputData[k]), string.Join(" ", outputData[k].Select(v => $"{Math.Round(v)} ~ {v:F6}")));
        }

        static void Main(string[] args)
        {
            string deviceType = "GPU";
            DeviceDescriptor device = null;
            if (deviceType == "GPU")
                device = DeviceDescriptor.GPUDevice(0);
            else
                device = DeviceDescriptor.CPUDevice;

            Console.WriteLine($"Device {device.AsString()} {device.Type}[{device.Id}]");
            Console.WriteLine();

            int inputSize = 2;
            int hiddenLayers = 4;
            int numClasses = 1;

            var inputs = CNTKLib.InputVariable(new int[] { inputSize }, DataType.Float, "features");
            var labels = CNTKLib.InputVariable(new int[] { numClasses }, DataType.Float, "labels");

            var MLPmodel = CreateModel(inputs, hiddenLayers, numClasses, device, "MLPmodel");
            var MLPtrainer = CreateModelTrainer(MLPmodel, inputs, labels);

            TrainFromArrays(MLPtrainer, inputs, labels, device);
            //TrainFromMiniBatchFile(MLPtrainer, inputs, labels, device);

            TestPrediction(MLPmodel, device);

            Console.WriteLine();
            Console.WriteLine("End");

            Console.ReadKey();
        }
    }
}
