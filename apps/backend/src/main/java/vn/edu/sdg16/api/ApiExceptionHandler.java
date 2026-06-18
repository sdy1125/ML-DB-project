package vn.edu.sdg16.api;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.ResourceAccessException;

@RestControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ResponseEntity<Map<String, Object>> validationError(
            MethodArgumentNotValidException exception
    ) {
        Map<String, String> fields = new LinkedHashMap<>();
        exception.getBindingResult().getFieldErrors().forEach(
                error -> fields.put(error.getField(), error.getDefaultMessage())
        );
        return ResponseEntity.badRequest().body(errorBody(
                HttpStatus.BAD_REQUEST,
                "Dữ liệu gửi lên không hợp lệ.",
                fields
        ));
    }

    @ExceptionHandler(RestClientResponseException.class)
    ResponseEntity<Map<String, Object>> upstreamError(
            RestClientResponseException exception
    ) {
        HttpStatus status = HttpStatus.resolve(exception.getStatusCode().value());
        HttpStatus resolved = status == null ? HttpStatus.BAD_GATEWAY : status;
        return ResponseEntity.status(resolved).body(errorBody(
                resolved,
                "Dịch vụ ML từ chối yêu cầu.",
                exception.getResponseBodyAsString()
        ));
    }

    @ExceptionHandler(ResourceAccessException.class)
    ResponseEntity<Map<String, Object>> serviceUnavailable(
            ResourceAccessException exception
    ) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(errorBody(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Không thể kết nối tới dịch vụ ML.",
                exception.getMostSpecificCause().getMessage()
        ));
    }

    private Map<String, Object> errorBody(
            HttpStatus status,
            String message,
            Object details
    ) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("timestamp", Instant.now());
        body.put("status", status.value());
        body.put("error", status.getReasonPhrase());
        body.put("message", message);
        body.put("details", details);
        return body;
    }
}
