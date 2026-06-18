package vn.edu.sdg16.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ml-service")
public record MlServiceProperties(
        String baseUrl,
        Duration connectTimeout,
        Duration readTimeout
) {
}
