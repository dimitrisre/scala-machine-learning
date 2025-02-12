import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.io.File
import scala.util.Random

object DeepLearningTrainingAndRun {

  def main(args: Array[String]): Unit = {
    //Dataset preparation
    val numEpochs = 15

    val labelIndex = 20531
    val numClasses = 5
    val batchSize = 64


    def readCSVDataset(csvFileClasspath:String, batchSize:Int, labelIndex:Int, numClasses:Int) = {
      val rr: RecordReader = new CSVRecordReader()
      val inputFile: File = new File(csvFileClasspath)
      rr.initialize(new FileSplit(inputFile))

      val iterator: DataSetIterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses)

      iterator
    }

    val trainingDatasetIt = readCSVDataset("data/output/gene_expression_cancer_rna/training_data.csv", batchSize, labelIndex, numClasses)
    trainingDatasetIt.setPreProcessor(new ReshapePreProcessor())

    val testDatasetIt = readCSVDataset("data/output/gene_expression_cancer_rna/test_data.csv", batchSize, labelIndex, numClasses)
    testDatasetIt.setPreProcessor(new ReshapePreProcessor())

    //Network hyper parameters
    val numInputs = labelIndex
    val numOfOutputs = numClasses
    val numHiddenNodes = 5000

    val layer_0 = new LSTM.Builder()
      .nIn(numInputs)
      .nOut(numHiddenNodes)
      .activation(Activation.RELU)
      .build()
    val layer_1 = new LSTM.Builder()
      .nIn(numHiddenNodes)
      .nOut(numHiddenNodes)
      .activation(Activation.RELU)
      .build()
    val layer_2= new RnnOutputLayer.Builder()
      .activation(Activation.SOFTMAX)
      .lossFunction(LossFunction.MCXENT)
      .nIn(numHiddenNodes)
      .nOut(numOfOutputs)
      .build()


    val LSTMConf: MultiLayerConfiguration =
      new NeuralNetConfiguration.Builder()
        .seed(Random.nextInt())
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
        .weightInit(WeightInit.XAVIER)
        .updater(new Adam(0.0002))
        .l2(1e-4)
        .list()
        .layer(0, layer_0)
        .layer(1, layer_1)
        .layer(2, layer_2)
        .build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(LSTMConf)
    model.init()

    model.setListeners(new ScoreIterationListener(1))

    val layers = model.getLayers
    var totalNumParams = 0L

    for (i <- 0 until layers.length) {
      val nParams = layers(i).numParams()
      println(s"Layer $i: $nParams parameters")
      totalNumParams += nParams
    }
    println(s"Total number of network parameters: $totalNumParams")

    println("Starting training...")
    for (j <- 0 until numEpochs) {
      println(s"Epoch $j started")
      model.fit(trainingDatasetIt)

      val eval: Evaluation = model.evaluate(testDatasetIt)
      println(eval.stats())

      trainingDatasetIt.reset()
      testDatasetIt.reset()

      println(s"Epoch $j finished")
    }

    println("Saving model to: models/cancer_prediction/model")
    model.save(new File("models/cancer_prediction/model"))
    println("Model Saved")
  }

  class ReshapePreProcessor() extends DataSetPreProcessor {
    override def preProcess(ds: api.DataSet): Unit = {
      val labelShape = ds.getLabels.shape()
      val featureShape = ds.getFeatures.shape()

      ds.setFeatures(ds.getFeatures.reshape(featureShape(0), featureShape(1), 1))
      ds.setLabels(ds.getLabels.reshape(labelShape(0), labelShape(1), 1))
    }
  }
}
