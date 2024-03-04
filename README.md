# Integration of Chicago and NYPD Crime Data
The project involves the integration of crime data from two major cities, Chicago and New York City, to facilitate comparative analysis and insights. This integration process is particularly challenging due to the differences in the legal frameworks and categorization of crimes in each city. Chicago operates under both federal and Illinois state laws, while New York City follows its own set of laws. To bridge this gap, the project employs BERT, a sophisticated Natural Language Processing (NLP) model, for sentiment analysis of the laws governing NYPD's operations. The objective is to map NYPD's laws to the closest corresponding laws in the Illinois Uniform Crime Reporting (IUCR) codes used by the Chicago Police Department (CPD).

# The data for this project comes from several sources:

- **Chicago Crime Data**: This dataset comprises crime reports in Chicago from 2001 to the present, detailing both arrests and non-arrest incidents.
                          https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data
- **NYPD Crime Data**: This dataset includes incidents for which the NYPD issued criminal court summons, reflecting minor legal violations.
                          https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Historic-/sv2w-rv3k/about_data
- **NYPD Arrest Data**: This dataset captures more severe incidents leading to arrests by the NYPD.
                          https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data
- **NYPD Shooting Data**: This dataset specifically focuses on shooting incidents, representing a subset of the most serious crimes.
                          https://data.cityofnewyork.us/Public-Safety/NYPD-Shooting-Incident-Data-Historic-/833y-fsy8/about_data
- **IUCR CPD Data**: This dataset outlines the Illinois Uniform Crime Reporting codes, a standardized set of crime categories used by the Chicago Police Department.
                          https://data.cityofchicago.org/Public-Safety/Chicago-Police-Department-Illinois-Uniform-Crime-R/c7ck-438e/about_data

To accomplish the goal of integrating these datasets, the project categorizes the data into two main types: crimes and arrests. The 'crimes' dataset includes general criminal incidents, while the 'arrests' dataset focuses on more severe offenses, such as shootings and murders. By using BERT to analyze and find the most closely related IUCR laws for every NYPD law, the project aims to create a unified framework that allows for a direct comparison of crime statistics between Chicago and New York City. This approach not only helps in understanding the similarities and differences in crime patterns between the two cities but also in standardizing crime reporting for broader analytical purposes.
