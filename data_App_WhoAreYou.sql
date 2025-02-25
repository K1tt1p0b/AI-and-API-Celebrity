-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: localhost    Database: db_miniprojectfinal
-- ------------------------------------------------------
-- Server version	9.0.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `age`
--

DROP TABLE IF EXISTS `age`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `age` (
  `age_ID` int NOT NULL AUTO_INCREMENT,
  `age_Date` date NOT NULL,
  `age_result` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`age_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `age`
--

LOCK TABLES `age` WRITE;
/*!40000 ALTER TABLE `age` DISABLE KEYS */;
INSERT INTO `age` VALUES (1,'2024-10-24','11.820123672485352'),(2,'2024-10-24','11.820123672485352'),(3,'2024-10-24','18.79496192932129'),(4,'2024-10-24','18.79496192932129'),(5,'2024-10-24','14.434134483337402'),(6,'2024-10-24','14.434134483337402'),(7,'2024-10-24','14.434134483337402'),(8,'2024-10-24','18.79496192932129'),(9,'2024-10-24','14.434134483337402'),(10,'2024-10-24','14.434134483337402'),(11,'2024-10-24','18.230087280273438'),(12,'2024-10-24','16.445838928222656'),(13,'2024-10-24','11.66201400756836'),(14,'2024-10-29','14'),(15,'2024-10-29','14'),(16,'2024-10-29','14'),(17,'2024-10-29','12'),(18,'2024-10-29','10'),(19,'2024-10-29','15'),(20,'2024-10-29','20'),(21,'2024-10-29','23'),(22,'2024-10-29','23'),(23,'2024-10-29','22'),(24,'2024-10-29','6'),(25,'2024-10-29','30'),(26,'2024-10-29','25'),(27,'2024-10-29','16'),(28,'2024-10-29','19'),(29,'2024-10-29','25'),(30,'2024-10-29','19');
/*!40000 ALTER TABLE `age` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `role`
--

DROP TABLE IF EXISTS `role`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `role` (
  `Role_ID` int NOT NULL AUTO_INCREMENT,
  `Type_Name` varchar(50) NOT NULL,
  PRIMARY KEY (`Role_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `role`
--

LOCK TABLES `role` WRITE;
/*!40000 ALTER TABLE `role` DISABLE KEYS */;
INSERT INTO `role` VALUES (1,'user'),(2,'admin');
/*!40000 ALTER TABLE `role` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `similarity`
--

DROP TABLE IF EXISTS `similarity`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `similarity` (
  `similarity_ID` int NOT NULL AUTO_INCREMENT,
  `similarity_Date` date NOT NULL,
  `similarityDetail_Percent` decimal(5,2) DEFAULT NULL,
  `ThaiCelebrities_ID` int DEFAULT NULL,
  `User_ID` int DEFAULT NULL,
  PRIMARY KEY (`similarity_ID`),
  KEY `ThaiCelebrities_ID` (`ThaiCelebrities_ID`),
  KEY `fk_User_ID` (`User_ID`),
  CONSTRAINT `fk_User_ID` FOREIGN KEY (`User_ID`) REFERENCES `users` (`Users_ID`),
  CONSTRAINT `similarity_ibfk_1` FOREIGN KEY (`ThaiCelebrities_ID`) REFERENCES `thaicelebrities` (`ThaiCelebrities_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=70 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `similarity`
--

LOCK TABLES `similarity` WRITE;
/*!40000 ALTER TABLE `similarity` DISABLE KEYS */;
INSERT INTO `similarity` VALUES (67,'2025-02-25',35.26,81,17),(68,'2025-02-25',35.26,81,17),(69,'2025-02-25',29.50,36,17);
/*!40000 ALTER TABLE `similarity` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `thaicelebrities`
--

DROP TABLE IF EXISTS `thaicelebrities`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `thaicelebrities` (
  `ThaiCelebrities_ID` int NOT NULL AUTO_INCREMENT,
  `ThaiCelebrities_name` varchar(255) NOT NULL,
  PRIMARY KEY (`ThaiCelebrities_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=101 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `thaicelebrities`
--

LOCK TABLES `thaicelebrities` WRITE;
/*!40000 ALTER TABLE `thaicelebrities` DISABLE KEYS */;
INSERT INTO `thaicelebrities` VALUES (1,'ซุปเปอร์บอน'),(2,'กลัฟ คณาวุฒิ'),(3,'กวาง อาริศา หอมกรุ่น'),(4,'ก้อย อรัชพร'),(5,'เก๋ไก๋ สไลเดอร์'),(6,'จ๊ะ นงมณี'),(7,'เจมส์ จิรายุ'),(8,'ชนาธิป สรงกระสินธ์'),(9,'เต๋อ ฉันทวิชช์'),(10,'ญาญ่า อุรัสยา'),(11,'เบลล่า ราณี'),(12,'เบสท์ คำสิงห์'),(13,'โบว์ กัญญารัตน์'),(14,'ใบเฟิร์น พิมพ์ชนก'),(15,'ปอป้อ ทรัพย์สิรี แต้รัตนชัย'),(16,'ฝ้าย 4EVE'),(17,'มาเบล PIXXE'),(18,'มาริโอ้ เมาเร่อ'),(19,'ลิซ่า'),(20,'ลำไย ไหทองคำ'),(21,'สไปร์ท SPD'),(22,'โอบ นิธิ วิวรรธนวรางค์'),(23,'ใหม่ดาวิกา'),(24,'อั้ม พัชราภา'),(25,'อิงโกะ PiXXiE'),(26,'โอปป้าทัชชี่'),(27,'ฮาร์ท ชุติวัฒน์ จันเคน'),(28,'เอวา'),(29,'อาจุมม่า'),(30,'วี วิโอเลต วอเทียร์'),(31,'วิน เมธวิน'),(32,'มิ้น ชาลิดา'),(33,'มายเมทเนท'),(34,'ฟาง-ธนันต์ธรญ์'),(35,'แพนเค้ก เขมนิจ'),(36,'นิกกี้ นฉัตร'),(37,'แต้ว ณฐพร'),(38,'ต้าเหนิง กัญญาวีร์ สองเมือง'),(39,'เจ้าขุน'),(40,'จิดาภา แช่มช้อย'),(41,'เจ้านาย วรรธนะสิน'),(42,'โดนัท ภัทรพลฒ์ เดชพงษ์วรานนท์'),(43,'เก้า สุภัสสรา'),(44,'ขวัญ อุษามณี'),(45,'คริษฐา สังสะโอภาส'),(46,'คริส หอวัง'),(47,'ครีมไลค์'),(48,'คารีสา สปริงเก็ตต์'),(49,'บิว วราภรณ์'),(50,'คิมเบอร์ลี่'),(51,'บอส ชนกันต์'),(52,'จินวุค คิม'),(53,'เจมี่ จุฑาพิชญ์'),(54,'เจแปน ภาณุพรรณ จันทนะวงษ์'),(55,'แจน พลอยชมพู'),(56,'ซ้อการ์ด'),(57,'ฐิสา วริฏฐิสา'),(58,'ณัฐทิชา จันทรวารีเลขา'),(59,'ณัฐรุจา ชุติวรรณโสภณ'),(60,'ณิชาภัทร ฉัตรชัยพลรัตน์'),(61,'บูม กฤติน'),(62,'มิกค์ ทองระย้า'),(63,'แบงค์ปิ'),(64,'เจเจ ชยกร'),(65,'โบว์ เมลดา'),(66,'ไบร์ท วชิรวิชญ์ ชีวอารี'),(67,'ปู ไปรยา'),(68,'พลอย หอวัง'),(69,'ปูเป้ เกศรินทร์'),(70,'พิ้งกี้ สาวิกา'),(71,'มิลิน ดอกเทียน'),(72,'พิมรี่พาย'),(73,'อแมนด้า ออบดัม'),(74,'ภีม วสุพล พรพนานุรักษ์'),(75,'อาเล็ก ธีรเดช เมธาวรายุทธ'),(76,'โอ๊ต ปราโมทย์'),(77,'ยูโร ยศวรรธน์'),(78,'สรยุทธ สุทัศนะจินดา'),(79,'สุภโชค สารชาติ'),(80,'นิว พีรดนย์'),(81,'แบงค์ ศุภณัฏฐ์'),(82,'ตะวันฉาย'),(83,'ดัง ณัฎฐ์ฐชัย'),(84,'ซุง ศตาวิน นาคทองเพชร'),(85,'ซิม คิวเท'),(86,'ชาริล ชับปุยส์'),(87,'ไทย ชญานนท์ ภาคฐิน'),(88,'มาร์ค กฤษณ์ กัญจนาทิพย์'),(89,'พีระกฤตย์ พชรบุณยเกียรติ'),(90,'กัน อรรถพันธ์'),(91,'โอห์ม ฐิติวัฒน์'),(92,'กาย ศิวกร'),(93,'ขุนพล ปองพล ปัญญามิตร'),(94,'เคน - ภูภูมิ พงศ์ภาณุ'),(95,'ฮง พิเชฐพงศ์'),(96,'ซี เดชชาติ'),(97,'เซียนหรั่ง'),(98,'เดรก สัตบุตร'),(99,'เอส ศุภ'),(100,'ไมค์ ภัทรเดช');
/*!40000 ALTER TABLE `thaicelebrities` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `Users_ID` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `Role_ID` int DEFAULT NULL,
  PRIMARY KEY (`Users_ID`),
  KEY `Role_ID` (`Role_ID`),
  CONSTRAINT `users_ibfk_1` FOREIGN KEY (`Role_ID`) REFERENCES `role` (`Role_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'admin','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(2,'pichai','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(17,'Test','$2b$12$WChZnogLJd8ZJArPkVBriu93FDp5tW1s.oNWaEVlJYcJ2L5KEM/Fe',1);
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping routines for database 'db_miniprojectfinal'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-02-25 20:58:02
