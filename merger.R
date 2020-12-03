### Load in Data ###
owid <- read.csv("data/owid.csv", sep=",")
cgrt <- read.csv("data/cgrt.csv", sep=",")
mobility_apple <- read.csv("data/mobility_apple_2.csv", sep=",")
mobility_google <- read.csv("data/Global_mobility_report.csv", sep=",")

### Specify countries of interest ###

iso_codes <- c("DEU", "AUT",
               "FRA", "ITA",
               "NLD", "GBR")

country_names <-c("Germany",
                  "Austria",
                  "France",
                  "Italy",
                  "Netherlands",
                  "United Kingdom")

### Specify variables of interest ###

columns_of_interest_owid <- c("iso_code",
                              "continent",
                              "date",
                              "location",
                              "new_deaths",
                              "new_cases",
                              "new_tests",
                              "population",
                              "life_expectancy"
                              )

columns_of_interest_apple <- c("geo_type",
                               "region",
                               "transportation_type",
                               "date",
                               "value")

areas_of_interest <- c("retail_and_recreation_percent_change_from_baseline",
                       "grocery_and_pharmacy_percent_change_from_baseline",
                       "parks_percent_change_from_baseline",
                       "transit_stations_percent_change_from_baseline",
                       "workplaces_percent_change_from_baseline" ,          
                       "residential_percent_change_from_baseline")

columns_of_interest_google <- c(areas_of_interest,
                                c("date",
                                  "country_region"))

columns_of_interest_cgrt <- c("C1_School.closing",
                              "C2_Workplace.closing",
                              "C3_Cancel.public.events",
                              "C4_Restrictions.on.gatherings",
                              "C4_Restrictions.on.gatherings",
                              "C5_Close.public.transport",
                              "C6_Stay.at.home.requirements",
                              "C7_Restrictions.on.internal.movement",
                              "C8_International.travel.controls",
                              "H1_Public.information.campaigns",
                              "H6_Facial.Coverings",
                              "E1_Income.support",
                              "CountryName",
                              "Date")


### Reshape Apple Mobility ###

# Drop countries not of interest
mobility_apple <- mobility_apple[mobility_apple$region %in%
                                   country_names &
                                   mobility_apple$geo_type == "country/region",
                                 !colnames(mobility_apple) %in%
                                   c("alternative_name",
                                     "sub.region",
                                     "country")]
# Specify time-varying variables
varying_vars <- colnames(mobility_apple)[!colnames(mobility_apple) %in% 
                                           c("geo_type",
                                             "region",
                                             "transportation_type")]
# Reshape from wide to long (temporary frame)
t <- reshape(mobility_apple,
             direction="long",
             v.names="value",
             idvar = c("geo_type",
                       "region",
                       "transportation_type"),
             varying = varying_vars)

# Re-assign dates
for (index in t$time) {
  date <- sub("X","",colnames(mobility_apple)[3+index])
  t$time[t$time == index] <- date
  
}
# Rename column and owerwrite original frame
colnames(t)[colnames(t) == "time"] <-"date"
mobility_apple <- t

### Clean up more recent google report ###
mobility_google <- mobility_google[mobility_google$country_region %in%
                                     country_names &
                                     mobility_google$sub_region_1 == "" &
                                     mobility_google$sub_region_2 == "",]


### reduce all frames to countries of interest to speed up later operations ###
# Then code date as date (first rename if appropriate)
# Rename country variables to always be "country"


owid_red <- owid[owid$iso_code %in% iso_codes,]
apple_red <- mobility_apple[mobility_apple$region %in% country_names,]
google_red <- mobility_google # already reduced
cgrt_red <- cgrt[cgrt$CountryName %in% country_names &
                   cgrt$Jurisdiction == "NAT_TOTAL",]

### Use Owid as origin (LHS for all joins) ###
owid_red$date <- as.Date(owid_red$date,"%Y-%m-%d")
owid_red <- owid_red[,columns_of_interest_owid]
colnames(owid_red)[colnames(owid_red) == "location"] <- "country"

data_raw_merged <- owid_red

### CGRT ###
cgrt_red$Date <- as.Date(as.character(cgrt_red$Date),"%Y%m%d")
cgrt_red <- cgrt_red[,columns_of_interest_cgrt]
colnames(cgrt_red)[colnames(cgrt_red) == "CountryName"] <- "country"
colnames(cgrt_red)[colnames(cgrt_red) == "Date"] <- "date"

data_raw_merged <- dplyr::left_join(data_raw_merged,
                                    cgrt_red,
                                    by=c("country","date"))

### Apple mobility ###
apple_red <- apple_red[,columns_of_interest_apple]
colnames(apple_red)[colnames(apple_red) == "region"] <- "country"
apple_red$date <- as.Date(as.character(apple_red$date),"%Y.%m.%d")
mobility_types = unique(apple_red$transportation_type)

for (mobility_type in mobility_types) {
  mobility_data <- apple_red[apple_red$transportation_type == mobility_type,c("country",
                                                                              "date",
                                                                              "value")]
  colnames(mobility_data)[colnames(mobility_data) == "value"] <- mobility_type
  data_raw_merged <- dplyr::left_join(data_raw_merged,
                                      mobility_data,
                                      by=c("country","date"))
  
}

### Google mobility ###
google_red <- google_red[,columns_of_interest_google]
colnames(google_red)[colnames(google_red) == "country_region"] <- "country"
google_red$date <- as.Date(as.character(google_red$date),"%Y-%m-%d")
data_raw_merged <- dplyr::left_join(data_raw_merged,
                                    google_red,
                                    by=c("country","date"))

write.csv(data_raw_merged, "data\\data_raw_merged.csv", row.names = FALSE)