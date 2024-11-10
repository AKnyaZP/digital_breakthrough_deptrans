package service

import (
	"context"
	"deptrans-backend/internal/repository"
)

type GeoDataService struct {
	geoDataRepository repository.GeoData
}

func NewGeoDataService(geoDataRepository repository.GeoData) *GeoDataService {
	return &GeoDataService{geoDataRepository: geoDataRepository}
}

func (s *GeoDataService) UploadData(ctx context.Context, dataSource string) {}
