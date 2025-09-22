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
-- Table structure for table `brand`
--

DROP TABLE IF EXISTS `brand`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `brand` (
  `brandID` int NOT NULL AUTO_INCREMENT,
  `brandName` varchar(255) NOT NULL,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`brandID`),
  UNIQUE KEY `brandName` (`brandName`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `brand`
--

LOCK TABLES `brand` WRITE;
/*!40000 ALTER TABLE `brand` DISABLE KEYS */;
INSERT INTO `brand` VALUES (1,'Dior','2025-07-31 09:42:02','2025-07-31 09:42:02'),(2,'MAC Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(3,'Maybelline','2025-07-31 09:42:02','2025-07-31 09:42:02'),(4,'L\'Oréal Paris','2025-07-31 09:42:02','2025-07-31 09:42:02'),(5,'Fenty Beauty','2025-07-31 09:42:02','2025-07-31 09:42:02'),(6,'Sephora Collection','2025-07-31 09:42:02','2025-07-31 09:42:02'),(7,'NARS Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(8,'Kylie Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02'),(9,'Anastasia Beverly Hills','2025-07-31 09:42:02','2025-07-31 09:42:02'),(10,'Benefit Cosmetics','2025-07-31 09:42:02','2025-07-31 09:42:02');
/*!40000 ALTER TABLE `brand` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `cosmetics`
--

DROP TABLE IF EXISTS `cosmetics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cosmetics` (
  `CosmeticID` int NOT NULL AUTO_INCREMENT,
  `Name` varchar(255) NOT NULL,
  `ShadeCode` varchar(50) DEFAULT NULL,
  `ShadeName` varchar(255) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Price` decimal(10,2) NOT NULL,
  `ImageURL` varchar(512) DEFAULT NULL,
  `ProductLink` varchar(512) DEFAULT NULL,
  `BrandID` int NOT NULL,
  `suitableSkinTone` varchar(50) DEFAULT NULL,
  `suitableBudgetRange` varchar(50) DEFAULT NULL,
  `suitableLookType` varchar(255) DEFAULT NULL,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`CosmeticID`),
  KEY `BrandID` (`BrandID`),
  CONSTRAINT `cosmetics_ibfk_1` FOREIGN KEY (`BrandID`) REFERENCES `brand` (`brandID`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cosmetics`
--

LOCK TABLES `cosmetics` WRITE;
/*!40000 ALTER TABLE `cosmetics` DISABLE KEYS */;
INSERT INTO `cosmetics` VALUES (1,'Dior Addict Lip Glow',NULL,NULL,'Lipstick',1500.00,'url_to_dior_lipglow_image.jpg','url_to_dior_lipglow_product.com',1,'Neutral','1500 - 3000','ธรรมชาติ','2025-07-31 13:10:02','2025-07-31 13:10:02');
/*!40000 ALTER TABLE `cosmetics` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `feedback`
--

DROP TABLE IF EXISTS `feedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `feedback` (
  `FeedbackID` int NOT NULL AUTO_INCREMENT,
  `CommentText` text,
  `Rating` int NOT NULL,
  `Date` date DEFAULT (curdate()),
  `UserID` int DEFAULT NULL,
  PRIMARY KEY (`FeedbackID`),
  KEY `fk_feedback_user` (`UserID`),
  CONSTRAINT `fk_feedback_user` FOREIGN KEY (`UserID`) REFERENCES `users` (`Users_ID`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `feedback`
--

LOCK TABLES `feedback` WRITE;
/*!40000 ALTER TABLE `feedback` DISABLE KEYS */;
INSERT INTO `feedback` VALUES (1,'แอปดีมากเลยค่ะ ชอบสุดๆ!',5,'2025-08-14',17),(2,'TEst',3,'2025-08-14',17);
/*!40000 ALTER TABLE `feedback` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `makeuplook`
--

DROP TABLE IF EXISTS `makeuplook`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `makeuplook` (
  `LookID` int NOT NULL AUTO_INCREMENT,
  `lookName` varchar(255) NOT NULL,
  `lookCategory` varchar(255) DEFAULT NULL,
  `description` text,
  PRIMARY KEY (`LookID`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `makeuplook`
--

LOCK TABLES `makeuplook` WRITE;
/*!40000 ALTER TABLE `makeuplook` DISABLE KEYS */;
INSERT INTO `makeuplook` VALUES (1,'ธรรมชาติ','Everyday Looks','ธรรมชาติใสๆ'),(2,'สายเกาหลี','Cultural Styles','เน้นผิวฉ่ำวาว อายไลเนอร์บางเบา สีปากสดใส'),(3,'สายฝอ','Cultural Styles','เน้นโครงหน้าชัด คอนทัวร์หนัก อายแชโดว์คมเข้ม'),(4,'สโมคกี้อายส์','Party Looks','เน้นดวงตาโดดเด่น ด้วยอายแชโดว์สีเข้ม'),(5,'Everyday Glam','Everyday Looks','ลุคที่ดูแต่งหน้า แต่ยังคงความเบาและสดใส เหมาะกับทุกวัน'),(6,'งานกลางคืน','Party Looks','ลุคสำหรับออกงานกลางคืน หรูหราและโดดเด่น'),(7,'วินเทจ','Thematic Looks','สไตล์การแต่งหน้าย้อนยุค เช่น ยุค 60s, 70s'),(8,'เทรนดี้','Seasonal Looks','ลุคที่กำลังอินเทรนด์ในปัจจุบัน'),(9,'งานรับปริญญา','Special Occasion','ลุคที่สุภาพ แต่ยังคงความสวยงาม เหมาะกับวันสำคัญ'),(10,'แนวพังก์','Alternative Styles','ลุคที่เน้นความขบถและมีเอกลักษณ์เฉพาะตัว');
/*!40000 ALTER TABLE `makeuplook` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `recommendedcolorpalettes`
--

DROP TABLE IF EXISTS `recommendedcolorpalettes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `recommendedcolorpalettes` (
  `PaletteID` int NOT NULL AUTO_INCREMENT,
  `PaletteName` varchar(255) NOT NULL,
  `SuitableSkinTone` varchar(50) NOT NULL,
  `ImageURL` varchar(512) NOT NULL,
  `Description` text,
  `createdAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`PaletteID`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `recommendedcolorpalettes`
--

LOCK TABLES `recommendedcolorpalettes` WRITE;
/*!40000 ALTER TABLE `recommendedcolorpalettes` DISABLE KEYS */;
INSERT INTO `recommendedcolorpalettes` VALUES (1,'Warm Tone General Color Palette','Warm Tone','warm.jpg','ตารางสีสำหรับผิวโทนอุ่น','2025-07-31 14:17:12','2025-07-31 15:33:56'),(2,'Cool Tone General Color Palette','Cool Tone','cool.jpg','ตารางสีสำหรับผิวโทนเย็น','2025-07-31 14:17:12','2025-07-31 15:33:56'),(3,'Neutral Tone General Color Palette','Neutral Tone','neutral.jpg','ตารางสีสำหรับผิวโทนกลาง','2025-07-31 14:17:12','2025-07-31 15:33:56');
/*!40000 ALTER TABLE `recommendedcolorpalettes` ENABLE KEYS */;
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
) ENGINE=InnoDB AUTO_INCREMENT=102 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `similarity`
--

LOCK TABLES `similarity` WRITE;
/*!40000 ALTER TABLE `similarity` DISABLE KEYS */;
INSERT INTO `similarity` VALUES (67,'2025-02-25',35.26,81,17),(68,'2025-02-25',35.26,81,17),(69,'2025-02-25',29.50,36,17),(70,'2025-02-25',35.26,81,17),(71,'2025-02-25',33.71,67,17),(72,'2025-03-05',29.88,26,17),(73,'2025-03-05',30.59,26,17),(74,'2025-03-05',30.34,26,17),(75,'2025-03-05',30.36,26,17),(76,'2025-07-30',57.98,47,17),(77,'2025-07-31',46.59,59,17),(78,'2025-07-31',46.59,59,17),(79,'2025-07-31',46.14,59,17),(80,'2025-07-31',45.28,59,17),(81,'2025-07-31',44.86,59,17),(82,'2025-08-12',46.68,59,17),(83,'2025-08-12',46.68,59,17),(84,'2025-08-12',46.49,59,17),(85,'2025-08-12',45.02,59,17),(86,'2025-08-12',44.70,59,17),(87,'2025-08-12',46.73,59,17),(88,'2025-08-14',56.92,48,17),(89,'2025-08-15',55.70,91,18),(90,'2025-08-15',63.87,17,18),(91,'2025-08-15',68.60,17,18),(92,'2025-08-15',68.60,17,17),(93,'2025-08-15',68.60,17,17),(94,'2025-08-15',68.60,17,17),(95,'2025-08-15',68.60,17,17),(96,'2025-08-15',68.60,17,17),(97,'2025-08-15',64.82,10,17),(98,'2025-08-18',57.15,61,18),(99,'2025-08-18',63.87,17,18),(100,'2025-08-18',55.39,38,19),(101,'2025-08-18',60.81,63,19);
/*!40000 ALTER TABLE `similarity` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `skintoneanalysis`
--

DROP TABLE IF EXISTS `skintoneanalysis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `skintoneanalysis` (
  `SkinToneAnalysisID` int NOT NULL AUTO_INCREMENT,
  `SkinTone` varchar(100) DEFAULT NULL,
  `Users_ID` int DEFAULT NULL,
  PRIMARY KEY (`SkinToneAnalysisID`),
  KEY `Users_ID` (`Users_ID`),
  CONSTRAINT `skintoneanalysis_ibfk_1` FOREIGN KEY (`Users_ID`) REFERENCES `users` (`Users_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=44 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `skintoneanalysis`
--

LOCK TABLES `skintoneanalysis` WRITE;
/*!40000 ALTER TABLE `skintoneanalysis` DISABLE KEYS */;
INSERT INTO `skintoneanalysis` VALUES (1,'Cool Tone',17),(2,'Cool Tone',17),(3,'Neutral Tone',17),(4,'Neutral Tone',17),(5,'Neutral Tone',17),(6,'Neutral Tone',17),(7,'Neutral Tone',17),(8,'Neutral Tone',17),(9,'Warm Tone',17),(10,'Warm Tone',17),(11,'Neutral Tone',17),(12,'Warm Tone',17),(13,'Neutral Tone',17),(14,'Neutral Tone',17),(15,'Neutral Tone',17),(16,'Warm Tone',17),(17,'Warm Tone',17),(18,'Warm Tone',17),(19,'Warm Tone',17),(20,'Warm Tone',17),(21,'Neutral Tone',17),(22,'Neutral Tone',17),(23,'Warm Tone',17),(24,'Neutral Tone',18),(25,'Neutral Tone',18),(26,'Neutral Tone',18),(27,'Neutral Tone',18),(28,'Neutral Tone',18),(29,'Neutral Tone',18),(30,'Cool Tone',18),(31,'Cool Tone',18),(32,'Cool Tone',18),(33,'Neutral Tone',17),(34,'Neutral Tone',17),(35,'Neutral Tone',17),(36,'Neutral Tone',18),(37,'Neutral Tone',18),(38,'Neutral Tone',18),(39,'Cool Tone',18),(40,'Cool Tone',18),(41,'Neutral Tone',19),(42,'Neutral Tone',19),(43,'Cool Tone',19);
/*!40000 ALTER TABLE `skintoneanalysis` ENABLE KEYS */;
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
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'admin','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(2,'pichai','$2b$12$6dbWIzTWcSD55qVzmNhPHOTTgO1j3xAPtctTdUqaMulw9glBbSGwa',2),(17,'Test','$2b$12$WChZnogLJd8ZJArPkVBriu93FDp5tW1s.oNWaEVlJYcJ2L5KEM/Fe',1),(18,'Nack','$2b$12$gAl8n6B8AXeYzcMva2lFJu54DbIrapNgWYi1anXcdgo1zA5.Y7/re',1),(19,'duangjai','$2b$12$ao3JXzw7UwK4Z/trDUsV7ezLqHZa7Hwp8sZxyD40KOK8syYFx6/ie',1);
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

-- Dump completed on 2025-09-22 21:41:05
