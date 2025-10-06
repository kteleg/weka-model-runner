# Step 1: Choose a base image with Java
FROM openjdk:21-jdk-slim

# Step 2: Set a working directory inside the container
WORKDIR /app

# Step 3: Copy JAR into the container
COPY target/weka-model-runner.jar ./weka-model-runner.jar

# Copy datasets to the container
COPY src/main/resources/*.csv ./datasets/

# Step 4: Default command to run your JAR
ENTRYPOINT ["java", "-jar", "weka-model-runner.jar"]