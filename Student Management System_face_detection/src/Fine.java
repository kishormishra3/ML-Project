package front;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

public class Fine implements ActionListener{

	
	JFrame frame6;
	JLabel l1,l2,l3,l4,/*l5,l6,l7,*/l8,l9,l10/*,l11,l12,l13,l14,l15,l16*/,la,la2;
	JButton b1;
	JButton b2;
	JPanel p1,p2,p3,p4;
	Fine()
	{
		frame6 = new JFrame("FINE");
		frame6.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		l1 = new JLabel("<html><span style='font-size:25px; color:Red;'>"+"Fine Records"+"</span></html>");
		la=new JLabel("Admission Number:");
		la2=new JLabel("     ");
		l2=new JLabel("Debarred");
		l2.setBorder(BorderFactory.createLineBorder(Color.black));
		l3=new JLabel("Back Paper");
		l3.setBorder(BorderFactory.createLineBorder(Color.black));
		l4=new JLabel("Fees Balance");
		l4.setBorder(BorderFactory.createLineBorder(Color.black));
		/*l15=new JLabel("             ");
		l15.setBorder(BorderFactory.createLineBorder(Color.black));
		l5=new JLabel("             ");
		l5.setBorder(BorderFactory.createLineBorder(Color.black));
		l6=new JLabel("             ");
		l6.setBorder(BorderFactory.createLineBorder(Color.black));
		l7=new JLabel("             ");
		l7.setBorder(BorderFactory.createLineBorder(Color.black));*/
		
		l8=new JLabel ("           ");
		l8.setBorder(BorderFactory.createLineBorder(Color.black));
		l9=new JLabel ("          ");
		l9.setBorder(BorderFactory.createLineBorder(Color.black));
		l10=new JLabel("          ");
		l10.setBorder(BorderFactory.createLineBorder(Color.black));
		/*l16=new JLabel("          ");
		l16.setBorder(BorderFactory.createLineBorder(Color.black));
		l11=new JLabel("          ");
		l11.setBorder(BorderFactory.createLineBorder(Color.black));
		l12=new JLabel("          ");
		l12.setBorder(BorderFactory.createLineBorder(Color.black));
		l13=new JLabel("          ");
		l13.setBorder(BorderFactory.createLineBorder(Color.black));
		l14=new JLabel("          ");
		l14.setBorder(BorderFactory.createLineBorder(Color.black));*/
		
		p1= new JPanel(new FlowLayout());
		p2= new JPanel(new GridLayout(3,2));
		p3= new JPanel(new FlowLayout());
		p4= new JPanel(new FlowLayout());
		
		//p1.setBorder(BorderFactory.createLineBorder(Color.black));
		p2.setBorder(BorderFactory.createLineBorder(Color.black));
		p3.setBorder(BorderFactory.createLineBorder(Color.black));
		
		b1= new JButton("Back");
		b2= new JButton("Home");
		b1.addActionListener(this);
		b2.addActionListener(this);
		
		l1.setHorizontalAlignment(JLabel.CENTER);
		l8.setHorizontalAlignment(JLabel.CENTER);
		l2.setHorizontalAlignment(JLabel.CENTER);
		l3.setHorizontalAlignment(JLabel.CENTER);
		l4.setHorizontalAlignment(JLabel.CENTER);
		/*l5.setHorizontalAlignment(JLabel.CENTER);
		l6.setHorizontalAlignment(JLabel.CENTER);
		l7.setHorizontalAlignment(JLabel.CENTER);
		l15.setHorizontalAlignment(JLabel.CENTER);
		
		*/
		p1.add(l1);
		p4.add(la);
		p4.add(la2);
		p2.add(l2);
		p2.add(l8);
		
		p2.add(l3);
		p2.add(l9);
		
		p2.add(l4);
		p2.add(l10);
		
		/*p2.add(l15);
		p2.add(l16);
		
		p2.add(l5);
		p2.add(l11);
		
		p2.add(l6);
		p2.add(l12);
		
		p2.add(l7);
		p2.add(l13);*/
		
				
		p3.add(b1);
		p3.add(b2);
		frame6.add(p1);
		frame6.add(p4);
		frame6.add(p2);
		frame6.add(p3);
		
		frame6.setLayout(new GridLayout(4,1));
		frame6.setResizable(false);
		frame6.setSize(600,500);
		frame6.setLocation(200, 100);
		frame6.setVisible(true);
	}
	public void actionPerformed(ActionEvent e)
	{
		if(e.getSource()==b1)
		{
			frame6.setVisible(false);
			new Option();
		}
		else if(e.getSource()==b2)
		{
			frame6.setVisible(false);
			new Start();
		}
	}
}
