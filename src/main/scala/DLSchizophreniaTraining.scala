import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.deeplearning4j.nn.api.{MaskState, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{InputPreProcessor, MultiLayerConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, DropoutLayer, LSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.common.primitives
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.{DataSet, api}
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.util
import java.io.File
import scala.jdk.CollectionConverters.seqAsJavaListConverter
import scala.util.Random

object DLSchizophreniaTraining {

  def main(args: Array[String]):Unit = {
    implicit val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("SchizophreniaModelTraining")
      .getOrCreate()

    val batchSize = 32

    val numOfEpochs = 80
    val numOfInputs = 18
    val numOFHiddenNodes = numOfInputs*3

    val trainingData = readParquet2("data/schizophrenia/training_data", batchSize)
    val testingData  = readParquet2("data/schizophrenia/testing_data", batchSize)

    val numOfOutcomes = trainingData.totalOutcomes()

    val layer_0 = new DenseLayer.Builder()
      .nIn(numOfInputs)
      .nOut(numOFHiddenNodes)
      .activation(Activation.RELU)
      .build()

    val layer_1 = new DenseLayer.Builder()
      .nIn(numOFHiddenNodes)
      .nOut(numOFHiddenNodes/2)
      .activation(Activation.RELU)
      .build()


    val outputLayer = new OutputLayer.Builder()
      .activation(Activation.SIGMOID)
      .lossFunction(LossFunction.XENT)
      .nIn(numOFHiddenNodes/2)
      .nOut(numOfOutcomes)
      .build()

    val multiLayerConfig: MultiLayerConfiguration =
      new NeuralNetConfiguration.Builder()
        .seed(Random.nextLong())
        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .weightInit(WeightInit.XAVIER)
        .updater(new Adam(0.00001))
        .l2(0.00001)
        .list()
        .layer(0, layer_0)
        .layer(1, layer_1)
        .layer(2, outputLayer)
        .build()


    val model: MultiLayerNetwork = new MultiLayerNetwork(multiLayerConfig)
    model.init()

    println(model.summary())
    println(s"Total outcomes: $numOfOutcomes")

//    model.setListeners(new ScoreIterationListener(100))

    for(j <-0 until numOfEpochs){
      println(s"Epoch $j started")
      model.fit(trainingData)

      val eval: Evaluation = model.evaluate(testingData)
      println(eval.stats())

      trainingData.reset()
      testingData.reset()
      println(s"Epoch $j finished")
    }

    spark.stop()

    val model_save_path = s"models/schizophrenia/model_${System.currentTimeMillis()}.zip"
    println(s"Saving model to: $model_save_path")
    ModelSerializer.writeModel(model, new File(s"$model_save_path"), true)
    println("Model Saved")

  }

  def readCSVDataset(csvFileClasspath:String, batchSize:Int, labelIndex:Int, numClasses:Int) = {
    val rr: RecordReader = new CSVRecordReader()
    val inputFile: File = new File(csvFileClasspath)
    rr.initialize(new FileSplit(inputFile))

    val iterator: DataSetIterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses)

    iterator
  }

  def readParquet2(path: String, batchSize: Int)(implicit spark: SparkSession): DataSetIterator = {
    val df = spark.read.parquet(path)
    val data = df.collect().map(row => (row.getAs[Vector]("features").toArray, row.getDouble(1)))

    val featureMatrix = Nd4j.create(data.map(_._1))
    val labelMatrix = Nd4j.create(data.map(_._2)).reshape(data.length, 1)

    val dataSet = new DataSet(featureMatrix, labelMatrix)

    val normalizer = new NormalizerStandardize()
    normalizer.fit(dataSet)

    new ListDataSetIterator(dataSet.asList(), batchSize)
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
