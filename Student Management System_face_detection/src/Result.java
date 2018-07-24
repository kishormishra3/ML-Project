package front;
import javax.swing.*;

import java.awt.*;
import java.awt.event.*;

public class Result implements ActionListener{

		JFrame frame5;
		JLabel l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,la,la2;
		JButton b1;
		JButton b2;
		JPanel p1,p2,p3,p4;
		Result()
		{
			frame5 = new JFrame("RESULT DETAILS");
			frame5.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			l1 = new JLabel("<html><span style='font-size:25px; color:Red;'>"+"Result Details"+"</span></html>");
			
			la=new JLabel("Roll Number:");
			la2=new JLabel("   1234  ");
			
			l2=new JLabel("SEMESTER");
			l2.setBorder(BorderFactory.createLineBorder(Color.black));
			l3=new JLabel("I");
			l3.setBorder(BorderFactory.createLineBorder(Color.black));
			l4=new JLabel("II");
			l4.setBorder(BorderFactory.createLineBorder(Color.black));
			l15=new JLabel("III");
			l15.setBorder(BorderFactory.createLineBorder(Color.black));
			l5=new JLabel("IV");
			l5.setBorder(BorderFactory.createLineBorder(Color.black));
			l6=new JLabel("V");
			l6.setBorder(BorderFactory.createLineBorder(Color.black));
			l7=new JLabel("VI");
			l7.setBorder(BorderFactory.createLineBorder(Color.black));
			
			l8=new JLabel ("CGPA");
			l8.setBorder(BorderFactory.createLineBorder(Color.black));
			l9=new JLabel ("          ");
			l9.setBorder(BorderFactory.createLineBorder(Color.black));
			l10=new JLabel("          ");
			l10.setBorder(BorderFactory.createLineBorder(Color.black));
			l16=new JLabel("          ");
			l16.setBorder(BorderFactory.createLineBorder(Color.black));
			l11=new JLabel("          ");
			l11.setBorder(BorderFactory.createLineBorder(Color.black));
			l12=new JLabel("          ");
			l12.setBorder(BorderFactory.createLineBorder(Color.black));
			l13=new JLabel("          ");
			l13.setBorder(BorderFactory.createLineBorder(Color.black));
			l14=new JLabel("          ");
			l14.setBorder(BorderFactory.createLineBorder(Color.black));
			
			p1= new JPanel(new FlowLayout());
			p2= new JPanel(new GridLayout(7,2));
			p3= new JPanel(new FlowLayout());
			p4= new JPanel(new FlowLayout());
			
			//p1.setBorder(BorderFactory.createLineBorder(Color.black));
			p2.setBorder(BorderFactory.createLineBorder(Color.black));
			p3.setBorder(BorderFactory.createLineBorder(Color.black));
			
			
			b1= new JButton("Back");
			b2= new JButton("Home");
			b1.addActionListener (this);
			b2.addActionListener (this);
			
			l1.setHorizontalAlignment(JLabel.CENTER);
			l8.setHorizontalAlignment(JLabel.CENTER);
			l2.setHorizontalAlignment(JLabel.CENTER);
			l3.setHorizontalAlignment(JLabel.CENTER);
			l4.setHorizontalAlignment(JLabel.CENTER);
			l5.setHorizontalAlignment(JLabel.CENTER);
			l6.setHorizontalAlignment(JLabel.CENTER);
			l7.setHorizontalAlignment(JLabel.CENTER);
			l15.setHorizontalAlignment(JLabel.CENTER);
			
			p1.add(l1);
			
			p4.add(la);
			p4.add(la2);
			
			p2.add(l2);
			p2.add(l8);
			
			p2.add(l3);
			p2.add(l9);
			
			p2.add(l4);
			p2.add(l10);
			
			p2.add(l15);
			p2.add(l16);
			
			p2.add(l5);
			p2.add(l11);
			
			p2.add(l6);
			p2.add(l12);
			
			p2.add(l7);
			p2.add(l13);
			
					
			p3.add(b1);
			p3.add(b2);
			frame5.add(p1);
			frame5.add(p4);
			frame5.add(p2);
			frame5.add(p3);
			
			frame5.setLayout(new GridLayout(4,1));
			frame5.setResizable(false);
			frame5.setSize(600,500);
			frame5.setLocation(200, 100);
			frame5.setVisible(true);
		}
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			if(e.getSource()==b1)
			{
				frame5.setVisible(false);
				new Option();
			}
			else if(e.getSource()==b2)
			{
				frame5.setVisible(false);
				new Start();
			}
		}

	}

	

