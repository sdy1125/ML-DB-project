package vn.edu.sdg16.api.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

public record SearchRequest(
        @NotBlank String query,
        @Min(1) @Max(20) int limit
) {
    public SearchRequest {
        if (limit == 0) {
            limit = 5;
        }
    }
}
