import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors

object GenerateComparisonData {
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

		// Remove headline, empty strings and abnormal coordinates such as [0.0,0.0] in raw data
		val YellowTaxiPickups = sc.textFile(yellowFile)
		.filter(str => str.length() != 0)
		.map(_.split(",")).map(fields => (fields(1), fields(5), fields(6)))
		.filter(t => t._2.toLowerCase != " pickup_longitude")
		.filter(t => t._2.toDouble != 0.0 && t._3.toDouble != 0.0)
		val YellowComparisonData = YellowTaxiPickups.map(t => Vectors.dense(t._2.toDouble, t._3.toDouble))
		YellowComparisonData.saveAsTextFile(DATA_HOME + "/YellowComparisonData")
		val GreenTaxiPickups = sc.textFile(greenFile)
		.filter(str => str.length() != 0)
		.map(_.split(","))
		.map(fields => (fields(1), fields(5), fields(6)))
		.filter(t => t._2 != "Pickup_longitude")
		.filter(t => t._2.toDouble != 0.0 && t._2.toDouble != 0.0)
		val GreenComparisonData = GreenTaxiPickups.map(t => Vectors.dense(t._2.toDouble, t._3.toDouble))
		GreenComparisonData.saveAsTextFile(DATA_HOME + "/GreenComparisonData")
		val UberPickups = sc.textFile(uberFile)
		.filter(str => str.length() != 0)
		.map(_.split(","))
		.map(fields => (fields(0), fields(2), fields(1)))
		.filter(t => t._2 != "\"Lon\"")
		val UberComparisonData = UberPickups.map(t => Vectors.dense(t._2.toDouble, t._3.toDouble))
		UberComparisonData.saveAsTextFile(DATA_HOME + "/UberComparisonData")
	}
}