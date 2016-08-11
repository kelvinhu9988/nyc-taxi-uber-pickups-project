import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// spark-submit --class "PickupsPrediction" ./target/scala-2.11/nyc-taxi-uber-pickups-project_2.11-1.0.jar

object PickupsPrediction {
	def main(args: Array[String]) {
		var date: String = "201404"
		if (args.length != 2) {
			Console.err.println("You must input two parameters separated with space: longituede latitude")
			System.exit(1)
		} 
		val lon: Double = args(0).toDouble
		val lat: Double = args(1).toDouble

		// Specify the path to local data files
		val DATA_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/data"
		val OUT_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/output"

		val conf = new SparkConf().setMaster("local[8]").setAppName(s"NYC Taxi Uber Pickups Project")
		val sc = new SparkContext(conf)
		sc.setLogLevel("WARN")

		// Load prediction model
		val model_prediction = LogisticRegressionModel.load(sc, OUT_HOME + "/PredictionModel")

		val prediction: Double = model_prediction.predict(Vectors.dense(lon, lat))
		var result: String = "Null"
		if (prediction == 0.0) {
			result = "Yellow Taxi"
		} else if (prediction == 1.0) {
			result = "Green Taxi"
		} else if (prediction == 2.0) {
			result = "Uber"
		}

		Console.out.println("Prediction of point (" + lon + "," + lat + ") is: " + result)
		

	}
}