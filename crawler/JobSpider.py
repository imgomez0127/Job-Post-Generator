from .NetSpider import NetSpider
import json
class JobSpider(NetSpider):
    index_url = "https://www.indeed.com/jobs" 
    def __init__(self, spider_id, data_directory,queue_link):
        super(JobSpider,self).__init__(spider_id,data_directory,queue_link)
    def getJobAsDict(self):
        jobTitle = self._getTitle("h3", 
        {"class":"icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"})
        company = self._getAuthor("div",{"class":"icl-u-lg-mr--sm icl-u-xs-mr--xs"})
        body = self._getArticle("div",["p","li"],{"class":
        "jobsearch-JobComponent-description icl-u-xs-mt--md"})
        return {"jobTitle":jobTitle,"company":company,"body":body}
    def exportJobToJsonFile(self):
        jobDict = self.getJobAsDict()
        with open(self.data_directory+jobDict["company"].replace(" ", "_")+"_"+jobDict["jobTitle"].replace(" ", "_")+".json","w") as f:
            json.dump(jobDict,f)
        
        
if __name__ == "__main__":
    testurl = "https://www.indeed.com/cmp/AquaQ-Analytics/jobs/Financial-Software-Developer-cfbfc703d817f5c2?sjdu=QwrRXKrqZ3CNX5W-O9jEvWZePZcXeI16EUz3N-7_gls0faMYiZ14Q_dfPtRGb17rTa11QF39SCvd5tneudWYmSc8D61E0s02QCQdWxgC5rk&tk=1d5i427625igg803&adid=249400932&vjs=3"
    spider = JobSpider(1,"data/",testurl)
    print(spider.getJobAsDict())
    spider.exportJobToJsonFile()
