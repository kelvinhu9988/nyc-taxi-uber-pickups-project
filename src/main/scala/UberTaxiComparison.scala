import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// Run locally:
// spark-submit --class "UberTaxiComparison" ./target/scala-2.11/nyc-taxi-uber-pickups-project_2.11-1.0.jar

object UberTaxiComparison {
	def main(args: Array[String]) {
		// Specify the path to local data files
		val DATA_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/data"
		val OUT_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/output"
		val yellowFile = DATA_HOME + "/yellow_tripdata_2014-04.csv"
		val greenFile = DATA_HOME + "/green_tripdata_2014-04.csv"
		val uberFile = DATA_HOME + "/uber-raw-data-apr14.csv"

		val conf = new SparkConf().setMaster("local[8]").setAppName(s"NYC Taxi Uber Pickups Project")
		val sc = new SparkContext(conf)
		sc.setLogLevel("WARN")

		
		// Load 201404 model
		val model_201404 = LogisticRegressionModel.load(sc, OUT_HOME + "/LogisticRegressionWithLBFGSModel_201404")
		// Load 201408 model
		val model_201408 = LogisticRegressionModel.load(sc, OUT_HOME + "/LogisticRegressionWithLBFGSModel_201408")
		// Load labeled points saved using RDD[LabeledPoint].saveAsTextFile with the default number of partitions
		val yellow_test_data = MLUtils.loadVectors(sc, DATA_HOME + "/YellowComparisonData")
		val green_test_data  = MLUtils.loadVectors(sc, DATA_HOME + "/GreenComparisonData")
		
		val yellow_prediction_201404 = model_201404.predict(yellow_test_data)
		val yellow_prediction_201408 = model_201408.predict(yellow_test_data)
		val uber_pickups_201404 = yellow_prediction_201404.filter(double => double == 2.0).count
		val uber_pickups_201408 = yellow_prediction_201408.filter(double => double == 2.0).count
		val yellow_test_data_size = yellow_test_data.count
		Console.out.println("Prediction of 201404 Model:")
		println("Predicted Uber Pickups: " + uber_pickups_201404)
		println("Sample Size: " + yellow_test_data_size)
		println("Percentage: " + uber_pickups_201404/yellow_test_data_size.toDouble)		
		Console.out.println("Prediction of 201408 Model:")
		println("Predicted Uber Pickups: " + yellow_prediction_201408.filter(double => double == 2.0).count)
		println("Sample Size: " + yellow_test_data.count)
		println("Percentage: " + uber_pickups_201408/yellow_test_data_size.toDouble)


		
	}
}