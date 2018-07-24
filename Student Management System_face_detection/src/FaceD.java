package front;

import java.io.BufferedReader;
import java.io.InputStreamReader;

class FaceD
{
	FaceD()
	{
		try{
			
			ProcessBuilder pb = new ProcessBuilder("C:/Users/kishor/AppData/Local/conda/conda/envs/tensorflow/python.exe","D:/face.py");
			Process pp = pb.start();
			int c=0;
			BufferedReader bfr = new BufferedReader(new InputStreamReader(pp.getInputStream()));

			System.out.println(".........start   process.........");
			String line = "";
			while ((line = bfr.readLine()) != null){
					if(line.equalsIgnoreCase("5")){
						FaceUser.frame.setVisible(false);
						new Option();
						break;
					}
				if(c==-1)
				{
					
				}
			}
			}catch(Exception e)
			{
				System.out.println(e);
			}
	}
}
